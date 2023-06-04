"""SAC implementation for Pendulum-v1 environment.

Reference:
    - https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb
"""

import random

import gym
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
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers(x)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action: np.ndarray):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action


class DDPG:
    """Deep deterministic policy gradient method."""

    def __init__(
        self,
        env_name: str,
        hidden_dim: int = 256,
        value_learning_rate: float = 1e-3,
        policy_learning_rate: float = 1e-4,
        replay_buffer_size: int = 1_000_000,
        max_frames: int = 12_000,
        max_steps=500,
        batch_size: int = 128,
        gamma: float = 0.99,
        soft_tau: float = 1e-2,
        epsilon: float = 1e-6,
    ):
        self.env: gym.Env = NormalizedActions(gym.make(env_name))

        state_dim = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        self.action_noise = 0.1

        self.value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        ).to(DEVICE)
        self.target_value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        ).to(DEVICE)

        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        ).to(DEVICE)
        self.target_policy_net = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        ).to(DEVICE)

        self._copy_network(src_net=self.value_net, target_net=self.target_value_net, soft_tau=soft_tau)
        self._copy_network(src_net=self.policy_net, target_net=self.target_policy_net, soft_tau=soft_tau)

        self.value_criterion = nn.MSELoss()

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_learning_rate)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.max_frames = max_frames
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.epsilon = epsilon

    def _copy_network(
        self,
        src_net: nn.Module,
        target_net: nn.Module,
        soft_tau: float = 1.0,
    ) -> None:
        for param, target_param in zip(src_net.parameters(), target_net.parameters()):
            data = target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            target_param.data.copy_(data)

    def train(self) -> None:
        frame_idx = 0
        rewards = []

        while frame_idx < self.max_frames:
            state = self.env.reset()
            episode_reward = 0

            for _ in range(self.max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.ddpg_update()

                state = next_state
                episode_reward += float(reward)
                frame_idx += 1

                if done:
                    break
            rewards.append(episode_reward)
            print(episode_reward)

    def test(self, render: bool = False) -> float:
        """."""
        state = self.env.reset()
        total_reward = 0.0
        for _ in range(self.max_steps):
            if render:
                self.env.render(mode="human")
            action = self.get_action(state=state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            total_reward += float(reward)

            if done:
                break

        return total_reward

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """Get action."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action = self.policy_net(state_tensor)
        action = action.detach().cpu().numpy()[0, 0]
        action += self.action_noise * np.random.randn(self.env.action_space.shape[0])
        return action

    def ddpg_update(self) -> None:
        """Update network parameters."""
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(DEVICE)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self._copy_network(src_net=self.value_net, target_net=self.target_value_net, soft_tau=self.soft_tau)
        self._copy_network(src_net=self.policy_net, target_net=self.target_policy_net, soft_tau=self.soft_tau)


if __name__ == "__main__":
    ddpg = DDPG(env_name="Pendulum-v1")
    ddpg.train()
    print(ddpg.test(render=True))
