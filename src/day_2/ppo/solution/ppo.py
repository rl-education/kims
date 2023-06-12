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

    def make_mini_batch(self, mini_batch_size: int):
        start_idx = 0
        for _ in range(self.batch_size // mini_batch_size):
            mini_batch = self.buffer[start_idx: start_idx + mini_batch_size]
            state, action, log_action_prob, reward, next_state, done = zip(*mini_batch)
            yield (
                torch.tensor(state, dtype=torch.float),
                torch.tensor(action, dtype=torch.float),
                torch.stack(log_action_prob),
                torch.tensor(reward, dtype=torch.float),
                torch.tensor(next_state, dtype=torch.float),
                torch.tensor(done, dtype=torch.float),
            )
            start_idx += mini_batch_size

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
            batch_size: int ,
            mini_batch_size: int = 256,
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
        batch_buffer = BatchBuffer(self.batch_size)

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
                    self.env.render(mode="human")

                action = self.select_action(state=state)
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

    def ppo_update(self, batch_buffer: BatchBuffer) -> None:
        """Update model."""
        for _ in range(self.ppo_epochs):
            for mini_batch in self.make_mini_batch(batch_buffer):
                s, a,r, s_prime, done_mask, old_log_prob, td_target, advantage = self.calc_advantage(mini_batch)

                mu, std = self.policy_net(s, softmax_dim=1)
                dist = nn.distributions.Normal(mu, std)
                log_prob = dist.log_prob(a)
                ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target)

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
                self.optimization_step += 1

    def make_mini_batch(self, batch_buffer: BatchBuffer):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []
        for j in range(self.batch_size):
            for i in range(self.mini_batch_size):
                transition = batch_buffer.buffer[i]
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
                s, a, prob_a, r, s_prime, done = transition

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append([r])
                s_prime_lst.append(s_prime)
                prob_a_lst.append([prob_a])
                done_mask = 0 if done else 1
                done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)

            mini_batch = torch.tensor(s_batch, dtype=torch.float), torch.tensor(a_batch, dtype=torch.float), \
                torch.tensor(prob_a_batch, dtype=torch.float), torch.tensor(r_batch, dtype=torch.float), \
                torch.tensor(s_prime_batch, dtype=torch.float), torch.tensor(done_batch, dtype=torch.float),
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, old_log_prob, r, s_prime, done_mask = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.tau * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, old_log_prob, r, s_prime, done_mask, td_target, advantage))

        return data_with_adv


if __name__ == "__main__":
    SEED = 777
    set_seed(seed=SEED)
    sac = PPO(env_name="Pendulum-v1", log=False, seed=SEED)
    sac.train(max_steps=12_000)
    sac.test(n_episodes=3, render=True)
