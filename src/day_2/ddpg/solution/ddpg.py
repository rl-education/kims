"""DDPG implementation for Pendulum-v1 environment.

Reference:
    - https://github.com/higgsfield/RL-Adventure-2/blob/master/7.soft%20actor-critic.ipynb
"""

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
TENSORBOARD_DIR = Path(__file__).parent.parent / "runs"


def set_seed(seed: int = 777):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        soft_tau: float = 1e-2,
        batch_size: int = 128,
        gamma: float = 0.99,
        seed: int = 777,
        log: bool = False,
    ):
        self.env: gym.Env = NormalizedActions(gym.make(env_name))
        self.env.seed(seed)

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

        self.soft_target_update(src_net=self.value_net, target_net=self.target_value_net)
        self.soft_target_update(src_net=self.policy_net, target_net=self.target_policy_net)

        self.value_criterion = nn.MSELoss()

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_learning_rate)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.batch_size = batch_size
        self.gamma = gamma
        self.soft_tau = soft_tau

        self.log = log
        if self.log:
            self.logger = SummaryWriter(
                log_dir=TENSORBOARD_DIR / f"dddg-pendulum-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )

    def soft_target_update(
        self,
        src_net: nn.Module,
        target_net: nn.Module,
        soft_tau: float = 1.0,
    ) -> None:
        for param, target_param in zip(src_net.parameters(), target_net.parameters()):
            data = target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            target_param.data.copy_(data)

    def train(self, max_steps: int) -> None:
        rewards = []

        progress_bar = trange(max_steps)
        state = self.env.reset()
        returns = 0
        episode_idx = 0
        for step in progress_bar:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            self.replay_buffer.push(state, action, reward, next_state, done)
            if len(self.replay_buffer) > self.batch_size:
                self.ddpg_update()

            state = next_state
            returns += float(reward)
            if done:
                progress_bar.set_description(
                    f"[TRAIN] Episode {episode_idx} ({step} steps) reward: {returns:.02f}",
                )
                state = self.env.reset()
                returns = 0.0
                episode_idx += 1
            step += 1

            rewards.append(returns)

    def test(self, n_episodes: int, render: bool = False) -> None:
        """."""
        state = self.env.reset()
        returns = 0.0
        done = False
        for episode_idx in range(n_episodes):
            while not done:
                if render:
                    self.env.render(mode="human")

                action = self.select_action(state=state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                returns += float(reward)
            if self.log:
                self.logger.add_scalar("reward", returns, episode_idx)
            print(f"[TEST] Episode {episode_idx} reward: {returns}")

    def select_action(self, state: np.ndarray) -> np.ndarray:
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

        self.soft_target_update(
            src_net=self.value_net,
            target_net=self.target_value_net,
            soft_tau=self.soft_tau,
        )
        self.soft_target_update(
            src_net=self.policy_net,
            target_net=self.target_policy_net,
            soft_tau=self.soft_tau,
        )


if __name__ == "__main__":
    SEED = 777
    set_seed(SEED)
    ddpg = DDPG(env_name="Pendulum-v1", log=False)
    ddpg.train(max_steps=12_000)
    ddpg.test(n_episodes=1, render=True)
