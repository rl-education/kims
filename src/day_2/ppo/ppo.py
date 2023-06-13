import random
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
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


class ActorCritic(nn.Module):
    """Actor-critic network."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        log_std_range: tuple[int, int] = (-20, 2),
    ):
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_range

        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)

    def policy(self, intermediate: Tensor) -> tuple[Tensor, Tensor]:
        """Compute mean and standard deviation of action distribution."""
        intermediate = self.layer(intermediate)
        mean = 2 * torch.tanh(self.mean_layer(intermediate))
        log_std = self.log_std_layer(intermediate)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std.exp()

    def value(self, x: Tensor) -> Tensor:
        """Compute state value."""
        intermediate = self.layer(x)
        value = self.value_layer(intermediate)
        return value


class PPO:
    def __init__(
        self,
        env_name: str,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.9,
        tau: float = 0.9,
        clip_param: float = 0.2,
        n_epochs: int = 10,
        rollout_len: int = 1,
        batch_size=320,
        minibatch_size=32,
        seed: int = 777,
        log: bool = False,
    ):
        self.env = gym.make(env_name)
        self.env.seed(seed)

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.model = ActorCritic(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param
        self.n_epochs = n_epochs
        self.rollout_len = rollout_len
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.log = log
        if self.log:
            self.logger = SummaryWriter(
                log_dir=TENSORBOARD_DIR / f"ppo-pendulum-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            )

    def train(self, max_steps: int):
        """Train agent.

        Notes:
            1. Collect rollouts.
            2. Compute advantage from rollouts.
            3. Train network.
        """
        state = self.env.reset()
        returns = 0.0
        episode_idx = 0
        rollouts = []
        rollout = []
        done = False

        progress_bar = trange(max_steps)
        for step in progress_bar:
            # Collect rollouts
            for _ in range(self.rollout_len):
                action, log_prob = self.select_action(state)
                next_state, reward, done, _ = self.env.step([action.item()])

                rollout.append((state, action, reward / 10.0, next_state, log_prob.item(), done))
                if len(rollout) == self.rollout_len:
                    rollouts.append(rollout)
                    rollout = []

                state = next_state
                returns += reward

            if len(rollouts) == self.batch_size:
                # Compute advantage from rollouts
                batches = self.make_mini_batch(rollouts)
                batches = self.compute_advantage(batches)
                rollouts.clear()

                # Train network
                self.train_net(batches)

            if done:
                if self.log:
                    self.logger.add_scalar("train/episode_reward", returns, episode_idx)
                progress_bar.set_description(
                    f"[TRAIN] Episode {episode_idx} ({step} steps) reward: {returns:.02f}",
                )

                state = self.env.reset()
                episode_idx += 1
                returns = 0.0

        self.env.close()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action."""
        mean, std = self.model.policy(torch.from_numpy(state).float())
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_action_prob = dist.log_prob(action)
        return action, log_action_prob

    def compute_log_action_prob(self, state: Tensor, action: Tensor) -> Tensor:
        """Compute log action probability."""
        mean, std = self.model.policy(state)
        dist = torch.distributions.Normal(mean, std)
        log_action_prob = dist.log_prob(action)
        return log_action_prob

    def make_mini_batch(
        self,
        rollouts: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    ) -> list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Make torch mini batches from rollouts."""
        states, actions, rewards, next_states, log_action_probs, dones = [], [], [], [], [], []
        batch = []

        for _ in range(self.batch_size // self.minibatch_size):
            for _ in range(self.minibatch_size):
                rollout = rollouts.pop()
                state, action, reward, next_state, log_action_prob, done = zip(*rollout)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                log_action_probs.append(log_action_prob)
                dones.append(done)

            mini_batch = (
                torch.FloatTensor(states),
                torch.FloatTensor(actions).unsqueeze(1),
                torch.FloatTensor(rewards).unsqueeze(1),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones).unsqueeze(1),
                torch.FloatTensor(log_action_probs).unsqueeze(1),
            )
            batch.append(mini_batch)

        return batch

    def compute_advantage(
        self,
        batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]],
    ) -> list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """Compute advantage and make batch."""
        with_advantage = []
        for mini_batch in batch:
            state, action, reward, next_state, done, old_log_action_prob = mini_batch

            # TODO 1: Compute advantage
            # 1. Compute td_target for each state (hint: use batch)
            #    TD Target = r + gamma * V(s')
            # 2. Compute td_error for each state (hint: use batch)
            #    TD Error = TD Target - V(s)
            # 3. Compute advantage for each state (hint: solve from the end of the list)
            #    Advantage = (TD Error + gamma * tau * Advantage) * gamma * tau + TD Error ...

            with_advantage.append(
                (state, action, old_log_action_prob, td_target, advantage),
            )

        return with_advantage

    def train_net(self, batch):
        """Train network."""
        for _ in range(self.n_epochs):
            for mini_batch in batch:
                state, action, old_log_action_prob, td_target, advantage = mini_batch
                log_prob = self.compute_log_action_prob(state, action)

                # TODO 2: Compute ppo loss
                # 1. Compute importance sampling (ratio between new policy and old policy)
                # 2. Compute surrogate loss (importance sampling ratio * advantage)
                # 3. Compute clipped surrogate loss (clip: 1 - self.clip_param, 1 + self.clip_param)
                # 4. Compute value loss (mse between td_target and value)
                # 5. Compute ppo loss (clipped surrogate loss + value loss)

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()


if __name__ == "__main__":
    SEED = 777
    set_seed(SEED)
    ppo = PPO("Pendulum-v1", seed=SEED, log=False)
    ppo.train(max_steps=100_000)
