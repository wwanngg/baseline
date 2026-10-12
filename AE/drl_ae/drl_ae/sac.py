from .upstream import MLP
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class NumericalStabilityError(FloatingPointError):

    def __init__(self, message: str, diagnostics: dict | None=None):
        super().__init__(message)
        self.diagnostics = {}
        for key, value in (diagnostics or {}).items():
            number = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
            self.diagnostics[key] = number if math.isfinite(number) else None

@dataclass(frozen=True)
class SACConfig:
    hidden: tuple[int, ...] = (256, 256, 256)
    learning_rate: float = 0.0003
    gamma: float = 0.99
    tau: float = 0.05
    batch_size: int = 256
    replay_capacity: int = 1000000
    random_steps: int = 10000
    initial_temperature: float = 0.2
    learn_temperature: bool = True
    temperature_learning_rate: float | None = None
    min_temperature: float = 0.0
    max_temperature: float = 0.0
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    gradient_clip_norm: float = 0.0
    critic_loss_kind: str = 'mse'
    q_abs_limit: float = 0.0

    def __post_init__(self):
        values = (self.learning_rate, self.gamma, self.tau, self.initial_temperature)
        if not np.isfinite(values).all() or self.learning_rate <= 0 or self.initial_temperature <= 0:
            raise ValueError('Invalid SAC learning rate or temperature')
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError('Invalid SAC discount or smoothing coefficient')
        if not self.hidden or min(self.hidden) < 1 or min(self.batch_size, self.replay_capacity) < 1 or (self.random_steps < 0):
            raise ValueError('Invalid network or replay settings')
        if self.replay_capacity < self.batch_size:
            raise ValueError('Replay capacity must be >= batch size')
        if self.temperature_learning_rate is not None and (not math.isfinite(self.temperature_learning_rate) or self.temperature_learning_rate <= 0):
            raise ValueError('Invalid temperature learning rate')
        bounds = (self.min_temperature, self.max_temperature, self.log_std_min, self.log_std_max, self.gradient_clip_norm, self.q_abs_limit)
        if not np.isfinite(bounds).all() or min(self.min_temperature, self.max_temperature, self.gradient_clip_norm, self.q_abs_limit) < 0:
            raise ValueError('Invalid SAC stability bounds')
        if not -20 <= self.log_std_min < self.log_std_max <= 2:
            raise ValueError('log_std bounds must be ordered within [-20, 2]')
        if self.min_temperature and self.initial_temperature < self.min_temperature:
            raise ValueError('Initial temperature is below the configured minimum')
        if self.max_temperature and self.initial_temperature > self.max_temperature:
            raise ValueError('Initial temperature is above the configured maximum')
        if self.min_temperature and self.max_temperature and (self.min_temperature > self.max_temperature):
            raise ValueError('Temperature bounds are reversed')
        if self.critic_loss_kind not in ('mse', 'huber'):
            raise ValueError('critic_loss_kind must be mse or huber')

def mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) -> nn.Sequential:
    network = MLP(input_dim, output_dim, hidden).net
    for layer in network:
        if isinstance(layer, nn.ReLU):
            layer.inplace = False
    return network

class Actor(nn.Module):

    def __init__(self, state_dim: int, hidden: tuple[int, ...], log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        self.network = mlp(state_dim, hidden, 4)
        self.log_std_min, self.log_std_max = (log_std_min, log_std_max)
        self.sample_diagnostics = {}

    def forward(self, states):
        mean, log_std = self.network(states).chunk(2, dim=-1)
        return (mean, log_std.clamp(self.log_std_min, self.log_std_max))

    def sample(self, states):
        mean, log_std = self(states)
        if not torch.isfinite(mean).all() or not torch.isfinite(log_std).all():
            raise NumericalStabilityError('Nonfinite policy distribution')
        self.sample_diagnostics = {'policy_mean_abs_max': mean.detach().abs().max(), 'policy_log_std_min': log_std.detach().min(), 'policy_log_std_max': log_std.detach().max()}
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw = distribution.rsample()
        action = torch.tanh(raw)
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        log_probability = (distribution.log_prob(raw) - correction).sum(dim=-1, keepdim=True)
        return (action, log_probability)

class Critic(nn.Module):

    def __init__(self, state_dim: int, hidden: tuple[int, ...]):
        super().__init__()
        self.q1 = mlp(state_dim + 2, hidden, 1)
        self.q2 = mlp(state_dim + 2, hidden, 1)

    def forward(self, states, actions):
        inputs = torch.cat([states, actions], dim=-1)
        return (self.q1(inputs), self.q2(inputs))

class ReplayBuffer:

    def __init__(self, capacity: int, state_dim: int, chunk_size: int=1024):
        self.capacity, self.state_dim = (capacity, state_dim)
        self.chunk_size = min(chunk_size, capacity)
        self.chunks = {}
        self.position = self.size = 0

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, terminated):
        state, next_state = (np.asarray(state, np.float32), np.asarray(next_state, np.float32))
        action = np.asarray(action, np.float32)
        if state.shape != (self.state_dim,) or next_state.shape != state.shape or action.shape != (2,):
            raise ValueError('Replay transition dimensions do not match')
        if not all((np.isfinite(value).all() for value in (state, next_state, action, reward))):
            raise ValueError('Nonfinite replay transition')
        chunk, offset = divmod(self.position, self.chunk_size)
        if chunk not in self.chunks:
            length = min(self.chunk_size, self.capacity - chunk * self.chunk_size)
            self.chunks[chunk] = {'states': np.empty((length, self.state_dim), np.float32), 'actions': np.empty((length, 2), np.float32), 'rewards': np.empty((length, 1), np.float32), 'next_states': np.empty((length, self.state_dim), np.float32), 'terminated': np.empty((length, 1), np.float32)}
        data = self.chunks[chunk]
        for key, value in zip(data, (state, action, reward, next_state, float(terminated))):
            data[key][offset] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator, device: torch.device) -> dict:
        if self.size < batch_size:
            raise ValueError('Insufficient replay transitions')
        indices = rng.integers(self.size, size=batch_size)
        chunks, offsets = np.divmod(indices, self.chunk_size)
        return {key: torch.as_tensor(np.stack([self.chunks[int(chunk)][key][offset] for chunk, offset in zip(chunks, offsets)]), device=device) for key in ('states', 'actions', 'rewards', 'next_states', 'terminated')}

class SACAgent:

    def __init__(self, state_dim: int, config: SACConfig | None=None, device: str='cpu'):
        self.state_dim = state_dim
        self.config = config or SACConfig()
        self.device = torch.device(device)
        self.actor = Actor(state_dim, tuple(self.config.hidden), self.config.log_std_min, self.config.log_std_max).to(self.device)
        self.critic = Critic(state_dim, tuple(self.config.hidden)).to(self.device)
        self.target = deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)
        self.log_temperature = torch.tensor(math.log(self.config.initial_temperature), device=self.device, requires_grad=True)
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=self.config.temperature_learning_rate or self.config.learning_rate)
        self.target_entropy = -2.0
        self.updates = 0

    @property
    def temperature(self):
        return self.log_temperature.exp()

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool=False) -> np.ndarray:
        state = np.asarray(observation, dtype=np.float32)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError('Invalid policy observation')
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        action = torch.tanh(self.actor(state)[0]) if deterministic else self.actor.sample(state)[0]
        if not torch.isfinite(action).all():
            raise NumericalStabilityError('Nonfinite policy action')
        return action[0].cpu().numpy().astype(np.float32)

    def _bound_temperature(self):
        if not torch.isfinite(self.log_temperature).all():
            raise NumericalStabilityError('Nonfinite log temperature')
        low = math.log(self.config.min_temperature) if self.config.min_temperature else -math.inf
        high = math.log(self.config.max_temperature) if self.config.max_temperature else math.inf
        with torch.no_grad():
            self.log_temperature.clamp_(low, high)
        if not torch.isfinite(self.temperature).all():
            raise NumericalStabilityError('Temperature overflow')

    def _check_values(self, stage, values: dict, apply_q_limit=False):
        maxima = torch.stack([value.detach().abs().max() for value in values.values()])
        if not torch.isfinite(maxima).all():
            raise NumericalStabilityError(f'Nonfinite {stage}', dict(zip(values, maxima)))
        if apply_q_limit and self.config.q_abs_limit and torch.any(maxima > self.config.q_abs_limit):
            raise NumericalStabilityError(f'{stage} exceeds q_abs_limit={self.config.q_abs_limit}', dict(zip(values, maxima)))

    def _gradient_step(self, loss, parameters, optimizer, stage):
        self._check_values(f'{stage} loss', {f'{stage}_loss': loss})
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(parameters)
        try:
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip_norm or math.inf, error_if_nonfinite=True)
        except RuntimeError as error:
            raise NumericalStabilityError(f'Nonfinite {stage} gradient') from error
        optimizer.step()
        return norm.detach()

    def update(self, buffer: ReplayBuffer, rng: np.random.Generator) -> dict:
        self._bound_temperature()
        batch = buffer.sample(self.config.batch_size, rng, self.device)
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.sample(batch['next_states'])
            target_q1, target_q2 = self.target(batch['next_states'], next_actions)
            soft_value = torch.minimum(target_q1, target_q2) - self.temperature.detach() * next_log_prob
            target = batch['rewards'] + self.config.gamma * (1 - batch['terminated']) * soft_value
        q1, q2 = self.critic(batch['states'], batch['actions'])
        self._check_values('critic/target Q', {'critic_q1_abs_max': q1, 'critic_q2_abs_max': q2, 'bellman_target_abs_max': target}, apply_q_limit=True)
        td1, td2 = (q1.detach() - target, q2.detach() - target)
        loss_function = F.smooth_l1_loss if self.config.critic_loss_kind == 'huber' else F.mse_loss
        critic_loss = 0.5 * (loss_function(q1, target) + loss_function(q2, target))
        critic_norm = self._gradient_step(critic_loss, self.critic.parameters(), self.critic_optimizer, 'critic')
        self.critic.requires_grad_(False)
        try:
            actions, log_prob = self.actor.sample(batch['states'])
            policy_q1, policy_q2 = self.critic(batch['states'], actions)
            self._check_values('policy Q', {'policy_q1_abs_max': policy_q1, 'policy_q2_abs_max': policy_q2}, apply_q_limit=True)
            actor_loss = (self.temperature.detach() * log_prob - torch.minimum(policy_q1, policy_q2)).mean()
            actor_norm = self._gradient_step(actor_loss, self.actor.parameters(), self.actor_optimizer, 'actor')
        finally:
            self.critic.requires_grad_(True)
        temperature_loss = torch.zeros((), device=self.device)
        if self.config.learn_temperature:
            temperature_loss = -(self.log_temperature * (log_prob + self.target_entropy).detach()).mean()
            self._gradient_step(temperature_loss, [self.log_temperature], self.temperature_optimizer, 'temperature')
            self._bound_temperature()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.config.tau)
        self.updates += 1
        scalars = {'critic_loss': critic_loss.detach(), 'actor_loss': actor_loss.detach(), 'temperature_loss': temperature_loss.detach(), 'temperature': self.temperature.detach(), 'critic_q_mean': (q1.detach().mean() + q2.detach().mean()) / 2, 'critic_q_abs_max': torch.maximum(q1.detach().abs().max(), q2.detach().abs().max()), 'target_q_abs_max': target.abs().max(), 'td_abs_mean': (td1.abs().mean() + td2.abs().mean()) / 2, 'td_abs_max': torch.maximum(td1.abs().max(), td2.abs().max()), 'critic_grad_norm': critic_norm, 'actor_grad_norm': actor_norm, 'policy_entropy_estimate': -log_prob.detach().mean(), 'policy_saturation_fraction': (actions.detach().abs() > 0.98).float().mean(), **self.actor.sample_diagnostics}
        result = dict(zip(scalars, torch.stack(list(scalars.values())).cpu().tolist()))
        if not np.isfinite(list(result.values())).all():
            raise NumericalStabilityError('SAC produced nonfinite diagnostics', result)
        return result

    def save(self, path: Path, run_config: dict, training_steps: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + '.tmp')
        torch.save({'format': 'drl-ae-sac', 'state_dim': self.state_dim, 'sac_config': asdict(self.config), 'run_config': run_config, 'training_steps': training_steps, 'updates': self.updates, 'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 'target': self.target.state_dict(), 'log_temperature': self.log_temperature.detach().cpu(), 'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict(), 'temperature_optimizer': self.temperature_optimizer.state_dict()}, temporary_path)
        temporary_path.replace(path)

    @classmethod
    def load(cls, path: Path, device: str='cpu') -> tuple['SACAgent', dict]:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get('format') not in ('drl-ae-sac', 'drl-ae-sac-v2'):
            raise ValueError('Expected a DRL-AE SAC checkpoint; upstream DQN checkpoints are incompatible')
        agent = cls(checkpoint['state_dim'], SACConfig(**checkpoint['sac_config']), device)
        for module, key in ((agent.actor, 'actor'), (agent.critic, 'critic'), (agent.target, 'target')):
            module.load_state_dict(checkpoint[key])
        with torch.no_grad():
            agent.log_temperature.copy_(checkpoint['log_temperature'])
        agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        agent.temperature_optimizer.load_state_dict(checkpoint['temperature_optimizer'])
        agent.updates = int(checkpoint['updates'])
        return (agent, checkpoint)
