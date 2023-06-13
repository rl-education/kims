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
    """Multi layer perceptron network for deciding policy."""

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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward."""
        policy = self.layers(x)
        logits = nn.functional.log_softmax(policy, dim=-1)
        return logits


class ValueNetwork(nn.Module):
    """Multi layer perceptron network for estimating value."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        return self.layers(x)


class REINFORCE:
    """REINFORCE Trainer."""

    def __init__(
        self,
        env_name: str,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
        batch_size: int = 2,
        seed: int = 777,
        log: bool = True,
    ):
        self.env = gym.make(env_name)
        self.env.seed = seed

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.batch_size = batch_size

        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(DEVICE)

        self.value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(DEVICE)

        self.gamma = gamma
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)

        self.log = log
        if self.log:
            self.logger = SummaryWriter(
                log_dir=TENSORBOARD_DIR
                / f"reinforce-cartpole-baseline-{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
            )

    def train(self, n_episodes: int) -> None:
        """Train agent.

        Args:
            n_episodes (int): Number of episodes to train for.
        """
        progress_bar = trange(n_episodes)
        for episode_idx in progress_bar:
            episodes = [self.run_episode() for _ in range(self.batch_size)]
            batch = self.make_batch(episodes)
            self.update_batch(batch)

            # Log metrics
            returns = sum(batch.rewards) / self.batch_size
            if self.log:
                self.logger.add_scalar("train/episode_reward", returns, episode_idx)
            progress_bar.set_description(
                f"Episode {episode_idx}: Reward {float(returns):02f}",
            )

    def make_batch(self, episodes: list[Episode]) -> Episode:
        """Make batch of episodes."""
        batch = Episode()
        for episode in episodes:
            batch.states.extend(episode.states)
            batch.actions.extend(episode.actions)
            batch.log_action_probs.extend(episode.log_action_probs)
            batch.rewards.extend(episode.rewards)
            batch.dones.extend(episode.dones)
        return batch

    def test(self, n_episodes: int = 1, render: bool = False) -> None:
        """Test agent."""
        for episode_idx in range(n_episodes):
            episode = self.run_episode(render=render)
            if self.log:
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
            Compute policy by sampling from the normal distribution.
        """
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        logits = self.policy_net(state_tensor)

        action_distribution = torch.distributions.Categorical(logits=logits)
        action = action_distribution.sample()
        return action.numpy(), action_distribution.log_prob(action)

    def update_batch(self, batch: Episode) -> None:
        """Update policy and value networks."""
        returns = 0.0
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        for step in range(len(batch) - 2, -1, -1):
            returns = batch.rewards[step] + self.gamma * returns * (1 - batch.dones[step])
            value = self.update_value_net(states=batch.states[step], returns=returns)
            self.update_policy_net(
                log_action_probs=batch.log_action_probs[step],
                baseline=value,
                returns=returns,
            )

        self.value_optimizer.step()
        self.policy_optimizer.step()

    def update_value_net(self, states: np.ndarray, returns: float) -> Tensor:
        """Update the value network with baseline.

        Notes:
            loss = 0.5 * (G_t - baseline) ^ 2
            where G_t is the return from time step t.
            baseline is the value function.
        """
        value = self.value_net(torch.from_numpy(states))
        loss = 0.5 * (returns - value) ** 2
        loss.backward()
        return value.detach()

    def update_policy_net(self, log_action_probs: Tensor, baseline: Tensor, returns: float) -> None:
        """Update the policy network with baseline.

        Notes:
            loss = -alpha * (G_t - baseline) * gradient of ln(pi(a_t | s_t))
            where G_t is the return from time step t.
            baseline is a constant.
        """
        loss = -log_action_probs * (returns - baseline)
        loss.backward()


if __name__ == "__main__":
    SEED = 777
    set_seed(SEED)
    reinforce = REINFORCE(env_name="CartPole-v1", log=False, batch_size=4, seed=SEED)
    reinforce.train(n_episodes=500)
    reinforce.test(n_episodes=1, render=True)
