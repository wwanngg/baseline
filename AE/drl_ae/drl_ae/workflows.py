from pathlib import Path
import csv
import json
import shutil
import sys
from time import perf_counter
from uuid import uuid4
import numpy as np
from .config import device_name, make_env, seed_all, split, write_json
from .dataset import AE_ROOT, protect_output, read_image, scenes
from .rewards import reward_terms, RewardConfig
from .sac import NumericalStabilityError, ReplayBuffer, SACAgent, SACConfig

def rollout(env, agent: SACAgent | None, seed: int, policy: str, collect_trace: bool=False):
    observation, initial = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 100003)
    total, stable_at, success_at = (0.0, None, None)
    flicker_sum = boundary_steps = clipped_steps = 0
    trace = []
    for step in range(1, env.config.max_steps + 1):
        action = agent.act(observation, deterministic=True) if agent is not None else rng.uniform(-1, 1, 2).astype(np.float32)
        observation, reward, _, truncated, info = env.step(action)
        total += reward
        flicker_sum += info['flicker']
        boundary_steps += int(info.get('at_boundary', False))
        clipped_steps += int(info.get('action_clipped', False))
        if info['converged'] and stable_at is None:
            stable_at = step
        if info['success'] and success_at is None:
            success_at = step
        if collect_trace:
            trace.append({'policy': policy, 'episode_seed': seed, **info})
        if info['success'] or truncated:
            break
    row = {'policy': policy, 'scene': env.scene.name, 'group': 'indoor' if env.scene.name.startswith('indoor_') else 'outdoor', 'episode_seed': seed, 'steps': step, 'return': total, 'reward_per_frame': total / step, 'stable_converged': stable_at is not None, 'stable_convergence_step': stable_at, 'success': success_at is not None, 'successful_convergence_step': success_at, 'final_well_exposed': info['well_exposed'], 'final_brightness_distance': info['brightness_distance'], 'final_brightness_error': info['brightness_error'], 'final_mean_intensity': info['mean_intensity'], 'final_sobel_noise_proxy': info['sobel_noise_proxy'], 'mean_flicker': flicker_sum / step, 'boundary_fraction': boundary_steps / step, 'clipped_action_fraction': clipped_steps / step, 'final_exposure_us': info['exposure_us'], 'final_gain_db': info['gain_db'], 'selected_exposure_us': info['measured_exposure_us'], 'selected_gain_db': info['measured_gain_db'], 'source_filename': info['source_filename'], 'initial_exposure_us': initial['exposure_us'], 'initial_gain_db': initial['gain_db']}
    return (row, trace)

def aggregate(rows: list[dict]) -> list[dict]:
    summaries = []
    for group in ('all', 'indoor', 'outdoor'):
        for policy in sorted({r['policy'] for r in rows}):
            selected = [r for r in rows if r['policy'] == policy and (group == 'all' or r['group'] == group)]
            if not selected:
                continue
            successes = [r for r in selected if r['success']]
            stable = [r for r in selected if r['stable_converged']]
            record = {'group': group, 'policy': policy, 'episodes': len(selected), 'scenes': len({r['scene'] for r in selected}), 'success_rate': len(successes) / len(selected), 'stable_convergence_rate': len(stable) / len(selected), 'mean_successful_convergence_steps': float(np.mean([r['successful_convergence_step'] for r in successes])) if successes else None, 'successful_episodes': len(successes), 'mean_steps_to_done': float(np.mean([r['steps'] for r in selected])), 'mean_reward_per_frame': float(np.mean([r['reward_per_frame'] for r in selected])), 'mean_return': float(np.mean([r['return'] for r in selected])), 'mean_final_brightness_distance': float(np.mean([r['final_brightness_distance'] for r in selected])), 'mean_final_sobel_noise_proxy': float(np.mean([r['final_sobel_noise_proxy'] for r in selected]))}
            for field in ('final_brightness_error', 'mean_flicker', 'boundary_fraction', 'clipped_action_fraction'):
                values = [r[field] for r in selected if r.get(field) is not None]
                record[f'mean_{field}'] = float(np.mean(values)) if values else None
            summaries.append(record)
    return summaries

def validation_metrics(rows):
    summary = aggregate(rows)[0]
    return {'validation_reward_per_frame': summary['mean_reward_per_frame'], 'validation_success_rate': summary['success_rate'], 'validation_successful_episodes': summary['successful_episodes'], 'validation_episodes': summary['episodes'], 'validation_successful_convergence_steps': summary['mean_successful_convergence_steps'], 'validation_stable_rate': summary['stable_convergence_rate'], 'validation_mean_brightness_error': summary['mean_final_brightness_error'], 'validation_mean_sobel_noise_proxy': summary['mean_final_sobel_noise_proxy'], 'validation_mean_flicker': summary['mean_mean_flicker'], 'validation_boundary_fraction': summary['mean_boundary_fraction'], 'validation_clipped_action_fraction': summary['mean_clipped_action_fraction']}

def checkpoint_rank(metrics, criterion='success_then_brightness'):
    if criterion == 'reward':
        return (metrics['validation_reward_per_frame'], 0.0, 0.0)
    if criterion != 'success_then_brightness':
        raise ValueError('Unknown checkpoint selection criterion')
    return (metrics['validation_success_rate'], -metrics['validation_mean_brightness_error'], metrics['validation_reward_per_frame'])

def window_metrics(rows):
    result = {}
    for source, target in (('reward', 'reward_mean'), ('brightness_error', 'brightness_error_mean'), ('reward_brightness', 'reward_brightness_mean'), ('reward_flicker', 'reward_flicker_mean'), ('reward_noise', 'reward_noise_mean'), ('reward_clip', 'reward_clip_mean'), ('flicker', 'flicker_mean'), ('sobel_noise_proxy', 'sobel_noise_proxy_mean'), ('at_boundary', 'boundary_fraction'), ('gain_at_boundary', 'gain_boundary_fraction'), ('exposure_at_boundary', 'exposure_boundary_fraction'), ('action_clipped', 'clipped_action_fraction'), ('well_exposed', 'well_exposed_fraction')):
        result[target] = float(np.mean([row[source] for row in rows]))
    return result

def train(cfg: dict) -> Path:
    training, validation, evaluation = split(cfg)
    rng = seed_all(cfg['seed'])
    device = device_name(cfg['device'])
    config = SACConfig(**cfg['sac'])
    total_steps = int(cfg['train']['steps'])
    if total_steps <= max(config.random_steps, config.batch_size):
        raise ValueError('Training budget must exceed random warmup and batch size')
    envs = [make_env(p, cfg) for p in training]
    validation_envs = [make_env(p, cfg) for p in validation]
    state_dim = envs[0].state_dim
    agent = SACAgent(state_dim, config, device)
    replay = ReplayBuffer(config.replay_capacity, state_dim)
    base = protect_output(Path(cfg['train']['output']), Path(cfg['dataset']['root']))
    base.mkdir(parents=True, exist_ok=True)
    version = int(cfg.get('implementation_version', 2))
    run_dir = base / f"sac_v{version}_seed{cfg['seed']}_{uuid4().hex[:8]}"
    run_dir.mkdir()
    write_json(run_dir / 'config.json', cfg)
    write_json(run_dir / 'split.json', {'train': [p.name for p in training], 'validation': [p.name for p in validation], 'eval': [p.name for p in evaluation]})
    print(f'运行目录：{run_dir}\n设备：{device}；训练/验证/测试场景数={len(training)}/{len(validation)}/{len(evaluation)}', flush=True)
    env = envs[int(rng.integers(len(envs)))]
    observation, _ = env.reset(seed=int(rng.integers(2 ** 31)), options={'training': True, 'training_step': 0})
    best, best_step, best_validation, step = (None, None, None, 0)
    no_improvement = 0
    window = []
    started = interval_started = perf_counter()
    stop_reason = 'interrupted_or_error'
    numerical_error = None
    last_saved_step = 0
    agent.save(run_dir / 'policy_last.pt', cfg, 0)
    patience = int(cfg['train'].get('early_stopping_patience', 0))
    minimum_stop_steps = int(cfg['train'].get('early_stopping_min_steps', 0))
    criterion = cfg['train'].get('checkpoint_metric', 'reward')
    print(f'Training budget: {total_steps}; early stopping: ' + (f'enabled, patience={patience}, starts at step {minimum_stop_steps}' if patience else 'disabled'), flush=True)
    with (run_dir / 'train.jsonl').open('w', encoding='utf-8') as log:
        try:
            for step in range(1, total_steps + 1):
                action = rng.uniform(-1, 1, 2).astype(np.float32) if step <= config.random_steps else agent.act(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                replay.add(observation, action, reward, next_observation, terminated)
                window.append(info)
                observation = next_observation
                losses = agent.update(replay, rng) if step > config.random_steps and len(replay) >= config.batch_size else None
                if terminated or truncated:
                    env = envs[int(rng.integers(len(envs)))]
                    observation, _ = env.reset(seed=int(rng.integers(2 ** 31)), options={'training': True, 'training_step': step})
                if step % int(cfg['train']['log_every']) == 0:
                    record = {'step': step, 'replay_size': len(replay), 'curriculum_level': info['curriculum_level'], 'reward': reward, 'implementation_version': version, 'state_dim': state_dim, 'training_profile': cfg.get('training_profile', 'stable_static_sac'), 'critic_loss_kind': config.critic_loss_kind, 'scene': info['scene'], 'elapsed_seconds': perf_counter() - started, 'steps_per_second': len(window) / max(perf_counter() - interval_started, 1e-09), **window_metrics(window), **(losses or {})}
                    log.write(json.dumps(record, allow_nan=False) + '\n')
                    log.flush()
                    print(f"Step {step}/{total_steps}; reward={record['reward_mean']:.4f}; brightness_error={record['brightness_error_mean']:.4f}; boundary={record['boundary_fraction']:.1%}; updates={agent.updates}", flush=True)
                    window.clear()
                    interval_started = perf_counter()
                if step % int(cfg['train']['evaluate_every']) == 0 and agent.updates:
                    rows = [rollout(e, agent, cfg['seed'] + episode, 'SAC')[0] for e in validation_envs for episode in range(int(cfg['train']['validation_episodes_per_scene']))]
                    metrics = validation_metrics(rows)
                    rank = checkpoint_rank(metrics, criterion)
                    improved = best is None or rank > best
                    if improved:
                        best, best_step, best_validation = (rank, step, metrics)
                        no_improvement = 0
                        agent.save(run_dir / 'policy_best.pt', cfg, step)
                    else:
                        no_improvement = no_improvement + 1 if patience and step >= minimum_stop_steps else 0
                    log.write(json.dumps({'type': 'validation', 'step': step, **metrics, 'episodes': rows, 'checkpoint_criterion': criterion, 'best_step': best_step, 'early_stopping_enabled': bool(patience), 'no_improvement_validations': no_improvement}, allow_nan=False) + '\n')
                    log.flush()
                    write_csv(run_dir / 'validation_latest.csv', rows)
                    agent.save(run_dir / 'policy_last.pt', cfg, step)
                    last_saved_step = step
                    print(f"验证：成功={metrics['validation_successful_episodes']}/{metrics['validation_episodes']}; brightness_error={metrics['validation_mean_brightness_error']:.4f}; reward/frame={metrics['validation_reward_per_frame']:.4f}; best_step={best_step}", flush=True)
                    if patience and step >= minimum_stop_steps and (no_improvement >= patience):
                        stop_reason = 'validation_plateau'
                        print(f'早停：检查点排序未改善，连续验证次数={no_improvement} 次', flush=True)
                        break
            else:
                stop_reason = 'budget_completed'
        except NumericalStabilityError as error:
            stop_reason = 'numerical_instability'
            numerical_error = {'message': str(error), 'diagnostics': error.diagnostics, 'failed_step': step, 'last_saved_policy_step': last_saved_step}
            log.write(json.dumps({'type': 'numerical_instability', 'step': step, **numerical_error}, allow_nan=False) + '\n')
            log.flush()
            print(f'Numerical stop at step {step}: {error}; keeping policy_last from step {last_saved_step}', flush=True)
        finally:
            if stop_reason != 'numerical_instability':
                agent.save(run_dir / 'policy_last.pt', cfg, step)
                last_saved_step = step
            write_json(run_dir / 'training_status.json', {'implementation_version': version, 'requested_steps': total_steps, 'completed_steps': step, 'optimizer_updates': agent.updates, 'stop_reason': stop_reason, 'best_step': best_step, 'early_stopping_enabled': bool(patience), 'early_stopping_patience': patience, 'early_stopping_min_steps': minimum_stop_steps, 'no_improvement_validations': no_improvement, 'last_saved_policy_step': last_saved_step, 'training_profile': cfg.get('training_profile', 'stable_static_sac'), 'numerical_error': numerical_error, 'best_validation': best_validation})
    if numerical_error is not None:
        raise RuntimeError(f"Numerical instability; see {run_dir / 'training_status.json'}")
    print(f"检查点：{run_dir / 'policy_last.pt'}", flush=True)
    return run_dir / 'policy_last.pt'
