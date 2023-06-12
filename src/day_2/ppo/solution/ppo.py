"""PPO implementation for Pendulum-v1 environment."""

import random
from datetime import datetime
from pathlib import Path
from typing import Generator

import gym
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
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


class BatchBuffer:
    """Batch buffer for PPO."""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0

    def push(
            self,
            state: np.ndarray,
            action: np.ndarray,
            log_action_prob: Tensor,
            reward: np.float64,
            next_state: np.ndarray,
            done: bool,
    ):
        # GAE return, advantage will be filled later
        self.buffer.append((state, action, log_action_prob, reward, next_state, done))

    def clear(self):
        self.buffer = [None] * self.batch_size
        self.position = 0

    def full(self):
        return len(self.buffer) == self.batch_size

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return zip(*self.buffer)


class GaussianPolicyNetwork(nn.Module):
    """Multi layer perceptron network for deciding policy."""

    def __init__(
            self,
            state_dim: int,
            hidden_dim: int,
            action_dim: int,
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

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        intermediate = self.layers(x)

        mean = self.mean_linear(intermediate)
        log_std = self.log_std_linear(intermediate)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std.exp()


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

    def forward(self, state: Tensor) -> Tensor:
        """Forward."""
        return self.layers(state)


class PPO:
    """PPO method."""

    def __init__(
        self,
        env_name: str,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        batch_size: int = 10,
        mini_batch_size: int = 32,
        ppo_epochs: int = 10,
        gamma: float = 0.99,
        tau: float = 0.95,
        clip_param: float = 0.2,
        seed: int = 777,
        log: bool = False,
    ) -> None:
        self.env = gym.wrappers.RescaleAction(gym.make(env_name), min_action=-1.0, max_action=1.0)
        self.env.seed(seed)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.policy_net = GaussianPolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )

        self.value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        )

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Define hyperparameter for training
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param

        self.log = log
        if self.log:
            self.writer = SummaryWriter(
                log_dir=TENSORBOARD_DIR / f"ppo-pendulum-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )

    def train(self, max_steps: int) -> None:
        """Learn model with PPO."""
        progress_bar = trange(max_steps)
        state = self.env.reset()
        returns = 0.0
        episode_idx = 0
        batch_buffer = BatchBuffer(self.batch_size * self.mini_batch_size)

        for step in progress_bar:
            action, log_action_prob = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            batch_buffer.push(state, action, log_action_prob, reward, next_state, done)

            if batch_buffer.full():
                self.ppo_update(batch_buffer)
                batch_buffer.clear()

            state = next_state
            returns += float(reward)

            if done:
                progress_bar.set_description(
                    f"[TRAIN] Episode {episode_idx} ({step} steps) reward: {returns:.02f}",
                )
                state = self.env.reset()
                returns = 0.0
                episode_idx += 1

    def test(self, n_episodes: int, render: bool = False) -> None:
        """Test agent."""
        state = self.env.reset()
        returns = 0.0
        done = False
        for episode_idx in range(n_episodes):
            while not done:
                if render:
                    self.env.render()

                action, _ = self.select_action(state=state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                returns += float(reward)
            if self.log:
                self.logger.add_scalar("reward", returns, episode_idx)
            print(f"[TEST] Episode {episode_idx} reward: {returns}")

    def select_action(self, state: np.ndarray):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        mean, std = self.policy_net(state_tensor)

        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        return action.detach().cpu().numpy()[0], normal.log_prob(z)

    def compute_log_action_prob(self, state: Tensor, action: Tensor) -> Tensor:
        mean, std = self.policy_net(state)
        normal = torch.distributions.Normal(mean, std)
        log_action_prob = normal.log_prob(action)
        return log_action_prob.sum(axis=-1)

    def ppo_update(self, batch) -> None:
        """Update model."""
        state, action, old_log_action_prob, reward, next_state, done = batch

        states = torch.FloatTensor(state).to(DEVICE)
        actions = torch.FloatTensor(action).to(DEVICE)
        log_action_probs = torch.FloatTensor(old_log_action_prob).to(DEVICE)
        rewards = torch.FloatTensor(reward).to(DEVICE)
        next_states = torch.FloatTensor(next_state).to(DEVICE)
        dones = torch.FloatTensor(done).to(DEVICE)

        # Compute advantage
        gaes = self.compute_gae(states, rewards, next_states, dones)
        advantages = self.compute_advantage(states, gaes)

        # Update policy network
        for _ in range(self.ppo_epochs):
            for state, action, old_log_action_prob, gae, advantage in self.ppo_iter(self.mini_batch_size, states, actions, log_action_probs, gaes, advantages):
                new_log_action_prob = self.compute_log_action_prob(state, action)
                ratio = (new_log_action_prob - old_log_action_prob).exp().unsqueeze(1)
                clipped_ratio = ratio.clamp(min=1.0 - self.clip_param, max=1.0 + self.clip_param).unsqueeze(1)
                surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)
                policy_loss = -surrogate.mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Update value network
                value_loss = F.mse_loss(self.value_net(state), gae.detach())

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

    def compute_gae(self, state, reward, next_state, done):
        values = torch.FloatTensor(state).to(DEVICE)
        next_values = torch.FloatTensor(next_state).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        done = torch.FloatTensor(done).unsqueeze(1).to(DEVICE)

        gae = 0.0
        returns = []
        for step in reversed(range(len(reward))):
            delta = reward[step] + self.gamma * next_values[step] * (1 - done[step]) - values[step]
            gae = delta + self.gamma * self.tau * (1 - done[step]) * gae
            returns.insert(0, gae + values[step])
        return torch.stack(returns)

    def compute_advantage(self, state, gae):
        values = torch.FloatTensor(state).to(DEVICE)
        advantage = gae - values
        return advantage

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]



if __name__ == "__main__":
    agent = PPO(env_name="Pendulum-v1")
    agent.train(max_steps=1)
    agent.test(n_episodes=3, render=True)

