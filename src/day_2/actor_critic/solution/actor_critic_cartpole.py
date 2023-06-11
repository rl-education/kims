"""REINFORCE Trainer."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange

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


@dataclass
class Transition:
    """Episode of the environment."""

    state: np.ndarray | Tensor
    action: np.ndarray | Tensor
    log_action_prob: Tensor
    reward: float | Tensor
    next_state: np.ndarray | Tensor
    done: bool | Tensor

    def __len__(self) -> int:
        """Return the length of the episode."""
        return len(self.reward)


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

    def forward(self, x: Tensor) -> Tensor:
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


class ActorCritic:
    """Actor Critic Trainer."""

    def __init__(
        self,
        env_name: str,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        learning_rate: float = 0.001,
        log: bool = False,
        batch_size: int = 2,
        seed: int = 777,
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

        self.value_net = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ).to(DEVICE)

        self.gamma = gamma
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)

        self.batch_size = batch_size
        self.log = log
        if self.log:
            self.logger = SummaryWriter(
                log_dir=TENSORBOARD_DIR / f"actor_critic-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )

    def train(self, max_steps: int) -> None:
        """Train agent.

        Args:
            max_steps (int): Number of steps to train for.
        """
        returns = 0.0
        episode_idx = 0
        batch = []

        state = self.env.reset()
        progress_bar = trange(max_steps)
        for step in progress_bar:
            transition = self.step(state)
            state = transition.next_state
            returns += transition.reward

            if len(batch) == self.batch_size:
                batch_transitions = self.make_batch(batch)
                self.update_network(batch_transitions)
                batch = []
            else:
                batch.append(transition)

            # Logging at the end of each episode
            if transition.done:
                progress_bar.set_description(
                    f"Episode {episode_idx} ({step} steps): Reward {returns:02f}",
                )
                if self.log:
                    self.logger.add_scalar("train/_episode_reward", returns, episode_idx)

                state = self.env.reset()
                episode_idx += 1
                returns = 0.0

    def make_batch(self, batch: list[Transition]) -> Transition:
        """Make batch of transitions."""
        state = torch.FloatTensor([transition.state for transition in batch]).to(DEVICE)
        action = torch.LongTensor([[transition.action.tolist()] for transition in batch]).to(DEVICE)
        log_action_prob = (
            torch.stack([transition.log_action_prob for transition in batch]).to(DEVICE).reshape(-1, 1)
        )
        reward = torch.FloatTensor([[transition.reward] for transition in batch]).to(DEVICE)
        next_state = torch.FloatTensor([transition.next_state for transition in batch]).to(DEVICE)
        done = torch.FloatTensor([[transition.done] for transition in batch]).to(DEVICE)

        return Transition(
            state=state,
            action=action,
            log_action_prob=log_action_prob,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def test(self, n_episodes: int = 1, render: bool = False) -> None:
        """Test agent."""
        self.policy_net.eval()
        self.value_net.eval()

        for episode_idx in range(n_episodes):
            state = self.env.reset()
            done = False
            returns = 0.0
            while not done:
                transition = self.step(state, render=render)
                state = transition.next_state
                returns += transition.reward
                done = transition.done
            if self.log:
                self.logger.add_scalar("test/episode_reward", returns, episode_idx)
            print(f"[TEST] Episode {episode_idx} reward: {returns}")
        self.env.close()

    def step(self, state: np.ndarray, render: bool = False) -> Transition:
        """Take one step in the environment."""
        action, log_action_prob = self.select_action(state)
        next_state, reward, done, _ = self.env.step(action)
        if render:
            self.env.render()
        return Transition(
            state=state,
            action=action,
            log_action_prob=log_action_prob,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def select_action(self, state: np.ndarray) -> tuple[np.ndarray, Tensor]:
        """Compute the action and log policy probability.

        Notes:
            Compute policy by sampling from the normal distribution.
        """
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        logits = self.policy_net(state_tensor)

        action_distribution = torch.distributions.Categorical(logits=logits)
        action = action_distribution.sample()
        log_action_prob = action_distribution.log_prob(action)
        return action.numpy(), log_action_prob

    def update_network(self, transition: Transition) -> Tensor:
        """Update the policy and value networks."""
        td_error = self.compute_td_error(transition)
        self.update_value_network(td_error)
        self.update_policy_network(transition, td_error)
        return td_error

    def compute_td_error(self, transition: Transition) -> Tensor:
        """Compute TD Error.

        Notes:
            TD Error = r + gamma * V(s') - V(s)
        """
        value = self.value_net(transition.state)
        next_value = self.value_net(transition.next_state)
        td_target = transition.reward + self.gamma * next_value * (1 - transition.done)
        td_error = td_target.detach() - value
        return td_error

    def update_value_network(self, td_error: Tensor) -> None:
        """Update value network.

        Notes:
            Loss = TD Error^2
        """
        critic_loss = td_error.pow(2).mean()
        self.value_optimizer.zero_grad()
        critic_loss.backward()
        self.value_optimizer.step()

    def update_policy_network(self, transition: Transition, td_error: Tensor) -> None:
        """Update policy network.

        Notes:
            Loss = -log(pi) * TD Error
        """
        self.policy_optimizer.zero_grad()
        policy_loss = -(transition.log_action_prob * td_error.detach()).mean()
        policy_loss.backward()
        self.policy_optimizer.step()


if __name__ == "__main__":
    set_seed(777)
    reinforce = ActorCritic(env_name="CartPole-v1", log=True, seed=777, batch_size=4)
    reinforce.train(max_steps=100_000)
    reinforce.test(n_episodes=3, render=True)
