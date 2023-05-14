"""SAC implementation for Pendulum-v1 environment.

Reference:
    - https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb
"""

import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class ReplayBuffer:
    """Replay buffer for SAC."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    """Multi layer perceptron network for predicting state value."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.layers(state)


class SoftQNetwork(nn.Module):
    """Multi layer perceptron network for predicting action value."""

    def __init__(self, state_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim + num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward."""
        emb = torch.cat([state, action], dim=1)
        return self.layers(emb)


class PolicyNetwork(nn.Module):
    """Multi layer perceptron network for deciding policy."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_actions: int,
        log_std_range: tuple[int, int] = (-20, 2),
    ):
        super().__init__()

        self.log_std_min, self.log_std_max = log_std_range

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        intermediate = self.layers(x)

        mean = self.mean_linear(intermediate)
        log_std = self.log_std_linear(intermediate)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class NormalizedActions(gym.ActionWrapper):
    def action(self, action: np.ndarray):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return actions


class SAC:
    """Soft actor critic method."""

    def __init__(
        self,
        env_name: str,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        replay_buffer_size: int = 1_000_000,
        max_frames: int = 40_000,
        max_steps=500,
        batch_size: int = 128,
        gamma: float = 0.99,
        mean_lambda: float = 1e-3,
        std_lambda: float = 1e-3,
        z_lambda: float = 0.0,
        soft_tau: float = 1e-2,
        epsilon: float = 1e-6,
    ):
        self.env_name = env_name
        self.env: gym.Env = NormalizedActions(gym.make(self.env_name))
        state_dim = self.env.observation_space.shape[0]  # type: ignore
        num_actions = self.env.action_space.shape[0]  # type: ignore

        self.value_net = ValueNetwork(state_dim=state_dim, hidden_dim=hidden_dim).to(DEVICE)
        self.target_value_net = ValueNetwork(state_dim=state_dim, hidden_dim=hidden_dim).to(DEVICE)
        self._update_target_value_net()

        self.soft_q_net = SoftQNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        ).to(DEVICE)
        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        ).to(DEVICE)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.max_frames = max_frames
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.mean_lambda = mean_lambda
        self.std_lambda = std_lambda
        self.z_lambda = z_lambda
        self.soft_tau = soft_tau
        self.epsilon = epsilon

    def _update_target_value_net(self, soft_tau: float = 1.0) -> None:
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            data = target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            target_param.data.copy_(data)

    def train(self) -> None:
        frame_idx = 0
        rewards = []

        while frame_idx < self.max_frames:
            state, _ = self.env.reset()
            episode_reward = 0

            for _ in range(self.max_steps):
                action = self.get_action(state)
                next_state, reward, terminate, truncate, _ = self.env.step(action)
                done = terminate or truncate

                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.soft_q_update()

                state = next_state
                episode_reward += float(reward)
                frame_idx += 1

                if done:
                    break
            rewards.append(episode_reward)
            print(episode_reward)

    def test(self, render: bool = False) -> float:
        """."""
        render_mode = "human" if render else None
        env = gym.make(self.env_name, render_mode=render_mode)
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = self.get_action(state=state)
            next_state, reward, terminate, truncate, _ = env.step(action)
            done = terminate or truncate
            state = next_state
            total_reward += float(reward)

        return total_reward

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        mean, log_std = self.policy_net(state_tensor)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        return action.detach().cpu().numpy()[0]

    def evaluate(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evalute."""
        mean, log_std = self.policy_net(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def soft_q_update(self) -> None:
        """Update network parameters."""
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(DEVICE)

        expected_q_value = self.soft_q_net(state, action)
        expected_value = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.evaluate(state)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * self.gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = self.mean_lambda * mean.pow(2).mean()
        std_loss = self.std_lambda * log_std.pow(2).mean()
        z_loss = self.z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._update_target_value_net(soft_tau=self.soft_tau)


if __name__ == "__main__":
    sac = SAC(env_name="Pendulum-v1")
    sac.train()
    print(sac.test(render=True))
