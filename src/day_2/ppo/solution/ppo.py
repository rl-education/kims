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


class ActorCritic(nn.Module):
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

    def pi(self, x):
        x = self.layer(x)
        mu = 2 * torch.tanh(self.mean_layer(x))
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std.exp()

    def v(self, x):
        x = self.layer(x)
        v = self.value_layer(x)
        return v


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
        rollout_len: int = 3,
        batch_size=320,
        minibatch_size=32,
        seed: int = 777,
        log: bool = False,
    ):
        # self.env = gym.wrappers.RescaleAction(gym.make(env_name), min_action=-1.0, max_action=1.0)
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

    def train(self, n_episodes: int):
        score = 0.0
        print_interval = 20
        batch = []
        rollout = []

        for n_epi in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                for _ in range(self.rollout_len):
                    action, log_prob = self.select_action(state)
                    s_prime, r, done, info = self.env.step([action.item()])

                    rollout.append((state, action, r / 10.0, s_prime, log_prob.item(), done))
                    if len(rollout) == self.rollout_len:
                        batch.append(rollout)
                        rollout = []

                    state = s_prime
                    score += r

                if len(batch) == self.batch_size:
                    batch = self.make_batch(batch)
                    batch = self.calc_advantage(batch)
                    self.train_net(batch)
                    batch.clear()

            if self.log:
                self.logger.add_scalar("train/episode_reward", score, n_epi)

            if n_epi % print_interval == 0 and n_epi != 0:
                print(
                    "# of episode :{}, avg score : {:.1f}".format(
                        n_epi,
                        score / print_interval,
                    ),
                )
                score = 0.0

        self.env.close()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action."""
        mu, std = self.model.pi(torch.from_numpy(state).float())
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_action_prob = dist.log_prob(action)
        return action, log_action_prob

    def compute_log_action_prob(self, state: np.ndarray, action: np.ndarray) -> Tensor:
        """Compute log action probability."""
        mu, std = self.model.pi(state)
        dist = torch.distributions.Normal(mu, std)
        log_action_prob = dist.log_prob(action)
        return log_action_prob

    def make_batch(self, batch):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for _ in range(self.batch_size // self.minibatch_size):
            for _ in range(self.minibatch_size):
                rollout = batch.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition

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

            mini_batch = (
                torch.tensor(s_batch, dtype=torch.float),
                torch.tensor(a_batch, dtype=torch.float),
                torch.tensor(r_batch, dtype=torch.float),
                torch.tensor(s_prime_batch, dtype=torch.float),
                torch.tensor(done_batch, dtype=torch.float),
                torch.tensor(prob_a_batch, dtype=torch.float),
            )
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + self.gamma * self.model.v(s_prime) * done_mask
                delta = td_target - self.model.v(s)
            delta = delta.numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * self.tau * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

    def train_net(self, data):
        for i in range(self.n_epochs):
            for mini_batch in data:
                s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                log_prob = self.compute_log_action_prob(s, a)
                ratio = torch.exp(log_prob - old_log_prob)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.model.v(s), td_target)

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()


if __name__ == "__main__":
    SEED = 777
    set_seed(SEED)
    ppo = PPO("Pendulum-v1", seed=SEED, log=False)
    ppo.train(n_episodes=5000)
