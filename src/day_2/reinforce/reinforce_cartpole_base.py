"""REINFORCE Trainer."""

import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
TENSORBOARD_DIR = Path(__file__).parent.parent.parent / "runs"


def set_seed(seed: int = 777):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Episode:
    """Episode of the environment."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_action_probs: list[Tensor] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add(self, state: np.ndarray, action: int, action_log_prob: Tensor, reward: float, done: bool) -> None:
        """Add a transition to the episode."""
        self.states.append(state)
        self.actions.append(action)
        self.log_action_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self) -> int:
        """Return the length of the episode."""
        return len(self.rewards)


class PolicyNetwork(nn.Module):
    """Multi layer perceptron network for deciding gaussian policy."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        # TODO 1: Implement logits
        return logits


class REINFORCE:
    """REINFORCE Discrete Trainer."""

    def __init__(
        self,
        env_name: str = "CartPole-v1",
        hidden_dim: int = 128,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
        seed: int = 777,
        log: bool = False,
    ):
        self.env = gym.make(env_name)
        self.env.seed(seed)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(DEVICE)

        self.gamma = gamma
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.log = log
        if self.log:
            self.logger = SummaryWriter(
                log_dir=TENSORBOARD_DIR
                / f"reinforce-cartpole-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )

    def train(self, n_episodes: int) -> None:
        """Train agent.

        Args:
            n_episodes (int): Number of episodes to train for.
        """
        progress_bar = trange(n_episodes)
        for episode_idx in progress_bar:
            episode = self.run_episode()
            self.update_policy_net(episode)

            # Log metrics
            progress_bar.set_description(f"Episode {episode_idx} reward: {sum(episode.rewards)}")
            if self.log:
                self.logger.add_scalar("train/_episode_reward", sum(episode.rewards), episode_idx)

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

            action, action_log_prob = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.add(
                state,
                action,
                action_log_prob,
                float(reward),
                done,
            )
            state = next_state
        return episode

    def select_action(self, state: np.ndarray) -> tuple[int, Tensor]:
        """Compute the action and log policy probability.

        Notes:
            Compute gaussian policy by sampling from the normal distribution.
        """
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        # TODO 2: Compute the action and action log probability
        # Sample action from softmax of logits
        return action, action_log_prob

    def update_policy_net(self, episode: Episode) -> None:
        """Update the policy network.

        Notes:
            loss = -alpha * G_t * gradient of ln(pi(a_t | s_t))
            where G_t is the return from time step t.
        """
        self.optimizer.zero_grad()
        # TODO 3: Compute the REINFORCE loss
        self.optimizer.step()


if __name__ == "__main__":
    SEED = 777
    set_seed(SEED)
    reinforce = REINFORCE(env_name="CartPole-v1", log=True, seed=SEED)
    reinforce.train(n_episodes=3000)
    reinforce.test(n_episodes=1, render=True)
