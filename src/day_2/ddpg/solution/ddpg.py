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


class ReplayBuffer:
    """Replay buffer for SAC."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Push a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
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
        batch_size: int = 256,
        gamma: float = 0.99,
        seed: int = 777,
        log: bool = False,
    ):
        self.env: gym.Env = gym.wrappers.RescaleAction(gym.make(env_name), -1.0, 1.0)
        self.env.seed(seed)

        state_dim = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        self.action_noise = 0.1

        self.value_net = QNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        ).to(DEVICE)
        self.target_value_net = QNetwork(
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

    def train(self, max_steps: int) -> None:
        """Train the agent.

        Notes:
            1. The agent execute one step at a time.
            2. Store the transition to replay buffer.
            3. If the replay buffer has transition more than buffer_size, update the network with batch size.
        """
        progress_bar = trange(max_steps)
        state = self.env.reset()
        returns = 0
        episode_idx = 0

        for step in progress_bar:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            # Update if replay buffer has enough transitions
            if len(self.replay_buffer) > self.batch_size:
                self.ddpg_update()

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
            step += 1

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
            2. Get action from policy network.
            3. Add noise to action.
            4. Clip action to [-1.0, 1.0].
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action = self.policy_net(state_tensor)
        action = action.detach().cpu().numpy()[0, 0]
        action += self.action_noise * np.random.randn(self.env.action_space.shape[0])
        action = np.clip(action, -1.0, 1.0)
        return action

    def ddpg_update(self) -> None:
        """Update network parameters.

        Notes:
            1. Sample a batch of transitions from replay buffer.
            2. Compute the actor loss.
            2. Compute the critic loss.
            3. Update the actor network.
            4. Update the critic network.
            5. Update the target networks.
        """
        # Sample a batch of transitions from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        action = torch.FloatTensor(action).to(DEVICE)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(DEVICE)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(DEVICE)

        self.update_actor_network(state)
        self.update_critic_network(state, action, next_state, reward, done)

        # Soft target update
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

    def update_actor_network(self, state) -> None:
        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def update_critic_network(self, state, action, next_state, reward, done) -> None:
        # Compute td target
        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value

        # Compute critic loss
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

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
    set_seed(SEED)
    ddpg = DDPG(env_name="Pendulum-v1", log=False, batch_size=256, seed=SEED)
    ddpg.train(max_steps=12_000)
    ddpg.test(n_episodes=1, render=True)
