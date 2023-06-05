"""REINFORCE Trainer."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
TENSORBOARD_DIR = Path(__file__).parent.parent.parent / "runs"


@dataclass
class Episode:
    """Episode of the environment."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_action_probs: list[Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def add(self, state: np.ndarray, action: int, action_log_prob: Tensor, reward: float) -> None:
        """Add a transition to the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.log_action_probs.append(action_log_prob)
        self.rewards.append(reward)

    def __len__(self) -> int:
        """Return the length of the episode."""
        return len(self.rewards)


class PolicyNetwork(nn.Module):
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
        self.log_std_linear = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        intermediate = self.layers(x)

        mean = self.mean_linear(intermediate)
        log_std = self.mean_linear(intermediate)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std.exp()


class NormalizedActions(gym.ActionWrapper):
    def action(self, action: np.ndarray):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action


class REINFORCE:
    """REINFORCE Trainer."""

    def __init__(
        self,
        env_name: str,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
    ):
        self.env = NormalizedActions(gym.make(env_name))

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(DEVICE)

        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.logger = SummaryWriter(
            log_dir=TENSORBOARD_DIR / f"reinforce-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

    def train(self, n_episodes: int) -> None:
        """Train agent.

        Args:
            n_episodes (int): Number of episodes to train for.
        """
        for episode_idx in trange(n_episodes):
            episode = self.run_episode()
            loss = self.update_policy_net_with_normalized_rewards(episode)
            # loss = self.update_policy_net_with_baseline(episode)

            # Log metrics
            self.logger.add_scalar("train/_episode_reward", sum(episode.rewards), episode_idx)
            self.logger.add_scalar("train/_loss", loss, episode_idx)
            for step in range(len(episode)):
                self.logger.add_scalar(f"train/entropy-{step}", episode.log_action_probs[step], episode_idx)

    def test(self, n_episodes: int = 1, render: bool = False) -> None:
        """Test agent."""
        for episode_idx in range(n_episodes):
            episode = self.run_episode(render=render)
            self.logger.add_scalar("test/episode_reward", sum(episode.rewards), episode_idx)
            print(f"[TEST] Episode {episode_idx} reward: {sum(episode.rewards)}")

    def run_episode(self, render: bool = False) -> Episode:
        """Run one episode of the environment."""
        episode = Episode()
        state = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render(mode="human")

            action, action_log_prob = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.add(
                state,
                action,
                action_log_prob,
                float(reward),
            )
            state = next_state
        return episode

    def get_action(self, state: np.ndarray) -> tuple[int, Tensor]:
        """Compute the action and log policy probability.

        Notes:
            Compute policy by sampling from the normal distribution.
        """
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        mean, std = self.policy_net(state_tensor)

        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        return action.numpy(), normal.log_prob(z)

    def update_policy_net(self, episode: Episode) -> torch.Tensor:
        """Update the policy network.

        Notes:
            loss = -alpha * G_t * gradient of ln(pi(a_t | s_t))
            where G_t is the return from time step t.
        """
        returns = 0.0
        self.optimizer.zero_grad()
        for step in range(len(episode) - 2, -1, -1):
            returns = episode.rewards[step] + self.gamma * returns
            loss = -episode.log_action_probs[step] * returns
            loss.backward()
        self.optimizer.step()
        return loss

    def update_policy_net_with_normalized_rewards(self, episode: Episode) -> torch.Tensor:
        """Update the policy network.

        Notes:
            loss = -alpha * G'_t * gradient of ln(pi(a_t | s_t))
            where G_t is the return from time step t.
            G'_t = G_t - mean(G_t) / std(G_t)
        """
        returns = torch.as_tensor(episode.rewards)
        for step in range(len(episode) - 2, -1, -1):
            returns[step] = episode.rewards[step] + self.gamma * returns[step + 1]
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        self.optimizer.zero_grad()
        for log_prob, discounted_reward in zip(episode.log_action_probs, returns):
            loss = -log_prob * discounted_reward
            loss.backward()
        self.optimizer.step()
        return loss

    def update_policy_net_with_baseline(self, episode: Episode) -> torch.Tensor:
        """Update the policy network with baseline.

        Notes:
            loss = -alpha * (G_t - baseline) * gradient of ln(pi(a_t | s_t))
            where G_t is the return from time step t.
            baseline is a constant.
        """
        returns = 0.0
        baseline = 500
        self.optimizer.zero_grad()
        for step in range(len(episode) - 2, -1, -1):
            returns = episode.rewards[step] + self.gamma * returns
            loss = -episode.log_action_probs[step] * (returns - baseline)
            loss.backward()
        self.optimizer.step()
        return loss


if __name__ == "__main__":
    reinforce = REINFORCE(env_name="Pendulum-v1")
    reinforce.train(n_episodes=1000)
    reinforce.test(n_episodes=10, render=True)
