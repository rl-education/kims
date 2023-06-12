"""SAC implementation for Pendulum-v1 environment."""

import random
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


class ReplayBuffer:
    """Replay buffer for SAC."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.float64,
        next_state: np.ndarray,
        done: bool,
    ):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


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


class SoftQNetwork(nn.Module):
    """Multi layer perceptron network for predicting action value."""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """Forward."""
        emb = torch.cat([state, action], dim=1)
        return self.layers(emb)


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


class SAC:
    """Soft actor critic method."""

    def __init__(
        self,
        env_name: str,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        replay_buffer_size: int = 1_000_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        soft_tau: float = 1e-2,
        seed: int = 777,
        log: bool = False,
    ):
        self.env = gym.wrappers.RescaleAction(gym.make(env_name), min_action=-1, max_action=1)
        self.env.seed = seed
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        self.value_net = ValueNetwork(state_dim=state_dim, hidden_dim=hidden_dim).to(DEVICE)
        self.target_value_net = ValueNetwork(state_dim=state_dim, hidden_dim=hidden_dim).to(DEVICE)
        self.soft_target_update(
            src_net=self.value_net,
            target_net=self.target_value_net,
            soft_tau=1.0,
        )

        self.soft_q_net = SoftQNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(DEVICE)

        self.policy_net = GaussianPolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        ).to(DEVICE)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=learning_rate)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.batch_size = batch_size
        self.gamma = gamma
        self.soft_tau = soft_tau

        self.log = log
        if self.log:
            self.logger = SummaryWriter(
                log_dir=TENSORBOARD_DIR / f"sac-pendulum-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            )

    def train(self, max_steps: int) -> None:
        """Train the agent.

        Notes:
            1. The agent execute one step at a time.
            2. Store the transition to replay buffer.
            3. If the replay buffer has transition more than buffer_size, update the network with batch size.
        """
        progress_bar = trange(max_steps)
        state = self.env.reset()
        returns = 0.0
        episode_idx = 0

        for step in progress_bar:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            # Update if replay buffer has enough transitions
            if len(self.replay_buffer) > self.batch_size:
                self.sac_update()

            state = next_state
            returns += float(reward)

            # Log if episode ends
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

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action.

        Notes:
            1. Convert state to tensor.
            2. Get mean and std from policy network.
            3. Sample action from normal distribution.
            4. Clip action with range [-1.0, 1.0].
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        mean, std = self.policy_net(state_tensor)

        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        return action.detach().cpu().numpy()[0]

    def compute_log_action_prob(
        self,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute action and its log probability.

        Notes:
            1. Convert state to tensor.
            2. Get mean and std from policy network.
            3. Sample action from normal distribution.
            4. Clip action with range [-1.0, 1.0].
        """
        mean, log_std = self.policy_net(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z)
        return action, log_prob

    def sac_update(self) -> None:
        """Update network parameters."""
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(DEVICE)

        self.update_soft_q_net(state, action, reward, next_state, done)
        self.update_value_net(state)
        self.update_policy_net(state)

        self.soft_target_update(
            src_net=self.value_net,
            target_net=self.target_value_net,
            soft_tau=self.soft_tau,
        )

    def update_soft_q_net(
        self,
        state: Tensor,
        action: Tensor,
        reward: Tensor,
        next_state: Tensor,
        done: Tensor,
    ) -> None:
        """Update soft q network parameters.

        Notes:
            1. Compute expected q value.
            2. Compute target q value.
                Next q value: reward + (1 - done) * gamma * target_value
            3. Compute soft q loss.
                Soft q loss: (expected q value - next q value) ** 2
            4. Update soft q network parameters.
        """
        expected_q_value = self.soft_q_net(state, action)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * self.gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

    def update_value_net(self, state: Tensor) -> None:
        """Update value network parameters.

        Notes:
            1. Compute expected value.
            2. Compute next value.
                Next value: expected new q value - log probability
            3. Compute value loss.
                Value loss: (expected value - next value) ** 2
            4. Update value network parameters.
        """
        expected_value = self.value_net(state)
        new_action, log_prob = self.compute_log_action_prob(state)
        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def update_policy_net(self, state: Tensor) -> None:
        """Update policy network parameters.

        Notes:
            1. Compute expected new q value.
            2. Compute expected value.
            3. Compute log probability target.
                Log probability target: expected new q value - expected value
            4. Compute policy loss.
                Policy loss: log probability * (log probability - log probability target)
            5. Update policy network parameters.
        """
        new_action, log_prob = self.compute_log_action_prob(state)

        expected_new_q_value = self.soft_q_net(state, new_action)
        expected_value = self.value_net(state)
        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def soft_target_update(
        self,
        src_net: nn.Module,
        target_net: nn.Module,
        soft_tau: float = 1.0,
    ) -> None:
        """Soft update the target network parameters.

        Notes:
            target = (1 - soft_tau) * target + soft_tau * src
        """
        for param, target_param in zip(src_net.parameters(), target_net.parameters()):
            data = target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            target_param.data.copy_(data)


if __name__ == "__main__":
    SEED = 777
    set_seed(seed=SEED)
    sac = SAC(env_name="Pendulum-v1", log=False, seed=SEED)
    sac.train(max_steps=12_000)
    sac.test(n_episodes=3, render=True)
