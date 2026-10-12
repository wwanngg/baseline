from __future__ import annotations
import argparse
import shutil
import sys
from collections import deque
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_exposure_rl.agents import DQNAgent, DQNConfig
from camera_exposure_rl.dataset import SceneGrid, discover_scenes, split_scenes
from camera_exposure_rl.env import ExposureEnv, EnvConfig, N_ACTIONS
from camera_exposure_rl.rewards import RewardConfig
from camera_exposure_rl.state import StateConfig
from camera_exposure_rl.utils import JsonlLogger, load_config, make_run_dir, resolve_device, seed_everything

def _build_envs(cfg: dict, paths: list[Path]) -> list[ExposureEnv]:
    state_cfg = StateConfig(roi_height=cfg['state']['roi']['height'], roi_width=cfg['state']['roi']['width'], resize_width=cfg['state']['resize_width'], history=cfg['state']['history'])
    reward_cfg = RewardConfig(brightness_target=cfg['reward']['brightness_target'], brightness_p=cfg['reward']['brightness_p'], w_mean=cfg['reward']['weights']['mean'], w_flicker=cfg['reward']['weights']['flicker'], w_sharpness=cfg['reward']['weights']['sharpness'])
    env_cfg = EnvConfig(max_episode_steps=cfg['env']['max_episode_steps'], early_termination=cfg['env']['early_termination'], convergence_threshold=cfg['env']['convergence_threshold'], convergence_patience=cfg['env']['convergence_patience'])
    side = cfg['dataset']['side']
    return [ExposureEnv(SceneGrid(p, side=side), state_cfg, reward_cfg, env_cfg) for p in paths]

def evaluate(agent: DQNAgent, envs: list[ExposureEnv], episodes: int) -> dict:
    returns: list[float] = []
    convergence_steps: list[int] = []
    final_brightness: list[float] = []
    for ep in range(episodes):
        env = envs[ep % len(envs)]
        obs, _ = env.reset(seed=ep)
        ep_return = 0.0
        last_r_mean = float('inf')
        for _ in range(env.env_config.max_episode_steps):
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            last_r_mean = info['r_mean']
            if terminated or truncated:
                break
        returns.append(ep_return)
        convergence_steps.append(info['step'])
        final_brightness.append(last_r_mean)
    return {'eval/return_mean': float(np.mean(returns)), 'eval/return_std': float(np.std(returns)), 'eval/steps_mean': float(np.mean(convergence_steps)), 'eval/final_brightness_penalty': float(np.mean(final_brightness))}

def train(config_path: Path) -> Path:
    cfg = load_config(config_path)
    rng = seed_everything(cfg['seed'])
    device = resolve_device(cfg['device'])
    scenes = discover_scenes(Path(cfg['dataset']['root']))
    train_paths, eval_paths = split_scenes(scenes, cfg['dataset'].get('eval_scenes'))
    if not train_paths:
        raise RuntimeError('Train split is empty — check `dataset.eval_scenes` in the config.')
    if not eval_paths:
        eval_paths = train_paths[-1:]
    train_envs = _build_envs(cfg, train_paths)
    eval_envs = _build_envs(cfg, eval_paths)
    state_dim = train_envs[0].observation_space.shape[0]
    agent_cfg = cfg['agent']
    agent = DQNAgent(DQNConfig(state_dim=state_dim, n_actions=N_ACTIONS, hidden_sizes=tuple(agent_cfg['hidden_sizes']), gamma=agent_cfg['gamma'], lr=float(agent_cfg['lr']), batch_size=agent_cfg['batch_size'], buffer_size=agent_cfg['buffer_size'], warmup_steps=agent_cfg['warmup_steps'], target_update_every=agent_cfg['target_update_every'], epsilon_start=agent_cfg['epsilon_start'], epsilon_end=agent_cfg['epsilon_end'], epsilon_decay_steps=agent_cfg['epsilon_decay_steps'], grad_clip=agent_cfg['grad_clip']), device=device, rng=rng)
    paths = make_run_dir(Path(cfg['train']['output_dir']))
    shutil.copy(config_path, paths.config_dump)
    print(f'[训练] run dir: {paths.root}')
    print(f'[训练] device: {device}')
    print(f'[训练] {len(train_paths)} train scenes, {len(eval_paths)} eval scenes')
    logger = JsonlLogger(paths.log_file)
    total_steps = cfg['train']['total_steps']
    log_every = cfg['train']['log_every']
    eval_every = cfg['train']['eval_every']
    early_stopping_patience = int(cfg['train'].get('early_stopping_patience', 0))
    best_metric = float('-inf')
    best_step: int | None = None
    no_improve = 0
    recent_returns: deque[float] = deque(maxlen=20)
    env = train_envs[int(rng.integers(len(train_envs)))]
    obs, _ = env.reset(seed=int(rng.integers(2 ** 31)))
    ep_return = 0.0
    stop_early = False
    for step in range(1, total_steps + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.remember(obs, action, reward, next_obs, terminated)
        ep_return += reward
        obs = next_obs
        metrics = agent.update()
        if terminated or truncated:
            recent_returns.append(ep_return)
            ep_return = 0.0
            env = train_envs[int(rng.integers(len(train_envs)))]
            obs, _ = env.reset(seed=int(rng.integers(2 ** 31)))
        if step % log_every == 0:
            record = {'step': step, 'train/return_mean_20': float(np.mean(recent_returns)) if recent_returns else None, 'train/epsilon': agent.epsilon}
            if metrics is not None:
                record.update({f'train/{k}': v for k, v in metrics.items()})
            logger.log(record)
            print(f"[训练] step={step:>6d}  ret20={record['train/return_mean_20']}  eps={agent.epsilon:.3f}")
        if step % eval_every == 0:
            agent.q_net.eval()
            with torch.no_grad():
                eval_metrics = evaluate(agent, eval_envs, cfg['train']['eval_episodes'])
            agent.q_net.train()
            metric = float(eval_metrics['eval/return_mean'])
            improved = metric > best_metric
            if improved:
                best_metric = metric
                best_step = step
                no_improve = 0
                agent.save(paths.checkpoint_best)
            else:
                no_improve += 1
            eval_metrics.update({'best/return_mean': best_metric, 'best/step': best_step, 'best/no_improve_evals': no_improve})
            logger.log({'step': step, **eval_metrics})
            tag = ' [NEW BEST]' if improved else ''
            print(f'[评估]  step={step:>6d}  {eval_metrics}{tag}')
            if early_stopping_patience > 0 and no_improve >= early_stopping_patience:
                print(f'[训练] early stopping at step {step}: no improvement for {no_improve} evals (best at step {best_step}, return_mean={best_metric:.4f})')
                stop_early = True
                break
    agent.save(paths.checkpoint)
    logger.close()
    print(f'[训练] saved last checkpoint -> {paths.checkpoint}')
    if best_step is not None:
        print(f'[训练] saved best checkpoint -> {paths.checkpoint_best}  (step {best_step}, return_mean={best_metric:.4f})')
    if stop_early:
        print('[训练] training halted by early stopping.')
    return paths.checkpoint

def main() -> None:
    parser = argparse.ArgumentParser(description='Train the camera-exposure RL agent.')
    parser.add_argument('--config', type=Path, default=Path('configs/default.yaml'), help='Path to the YAML configuration file.')
    args = parser.parse_args()
    train(args.config)
if __name__ == '__main__':
    main()
