"""PPO implementation for Pendulum-v1 environment.

Reference:
    - https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
"""

from typing import Generator

import gymnasium as gym
import numpy as np
import torch
from icecream import ic
from torch import nn, optim
from torch.nn import functional as F
from tqdm import trange

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class ActorNetwork(nn.Module):
    """Actor Network which decide action."""

    def __init__(self, state_dim: int, action_num: int, hidden_size: int = 256):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_num),
        )

        self.log_std = nn.Parameter(torch.ones(1, action_num) * 0.0)

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal:
        """Forward and compute action distributions."""
        mu = self.layers(x)
        std = self.log_std.exp().expand_as(mu)
        action_distribution = torch.distributions.Normal(mu, std)
        return action_distribution


class CriticNetwork(nn.Module):
    """Critic Network which predice action value."""

    def __init__(self, state_dim: int, hidden_size: int = 256):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.layers(x)


class PPO:
    """PPO method."""

    def __init__(
        self,
        env_name: str,
        num_envs: int = 1,
        hidden_size: int = 256,
        learning_rate: float = 3e-4,
        num_steps: int = 2000000,
        batch_size: int = 4096 * 4,
        mini_batch_size: int = 256,
        ppo_epochs: int = 4,
        gamma: float = 0.99,
        tau: float = 0.95,
        clip_param: float = 0.2,
    ) -> None:
        # Create environments
        self.env_name = env_name
        self.env = gym.vector.make(self.env_name, num_envs=num_envs)
        self.state_dim = self.env.observation_space.shape[1]  # type: ignore
        self.action_num = self.env.action_space.shape[1]  # type: ignore

        # Create actor-critic network
        self.actor = ActorNetwork(
            state_dim=self.state_dim,
            action_num=self.action_num,
            hidden_size=hidden_size,
        ).to(DEVICE)

        self.critic = CriticNetwork(
            state_dim=self.state_dim,
            hidden_size=hidden_size,
        ).to(DEVICE)

        # Create an optimizer
        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": learning_rate},
                {"params": self.critic.parameters(), "lr": learning_rate},
            ],
        )

        # Define hyperparameter for training
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param

    def compute_gae(
        self,
        next_value: torch.Tensor,
        rewards: list[torch.Tensor],
        masks: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        values = values + [next_value]
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])

        return returns

    def ppo_iter(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantage: torch.Tensor,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """PPO iteration."""
        for _ in range(self.batch_size // self.mini_batch_size):
            rand_ids = torch.randint(0, self.batch_size, (self.mini_batch_size,))
            yield (
                states[rand_ids, :],
                actions[rand_ids, :],
                log_probs[rand_ids, :],
                returns[rand_ids, :],
                advantage[rand_ids, :],
            )

    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> None:
        """Update model."""
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(
                states,
                actions,
                log_probs,
                returns,
                advantages,
            ):
                dist = self.actor(state)
                value = self.critic(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surr2 *= advantage

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(return_, value)

                loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(
        self,
    ) -> None:
        """Learn model with PPO."""
        state, _ = self.env.reset()

        step = 0
        test_rewards = []
        while step < self.num_steps:
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            for _ in trange(self.batch_size):
                state = torch.FloatTensor(state)
                dist = self.actor(state)
                value = self.critic(state)

                action = dist.sample()
                next_state, reward, terminate, truncate, _ = self.env.step(action.cpu().numpy())
                done = np.any((terminate, truncate), axis=0)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1))

                states.append(state)
                actions.append(action)

                if done.all():
                    next_state, _ = self.env.reset()
                state = next_state
                step += 1

            next_state = torch.FloatTensor(next_state)  # type: ignore[reportUnboundVariable]
            next_value = self.critic(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values

            self.ppo_update(
                states,
                actions,
                log_probs,
                returns,
                advantage,
            )

            test_reward = np.mean([self.test() for _ in range(10)])
            test_rewards.append(test_reward)
            ic(test_reward)

    def test(self, render: bool = False) -> float:
        """."""
        render_mode = "human" if render else None
        env = gym.make(self.env_name, render_mode=render_mode)
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            dist = self.actor(state)
            action = dist.sample().cpu().numpy()[0]
            next_state, reward, terminate, truncate, _ = env.step(action)
            done = terminate or truncate
            state = next_state
            total_reward += float(reward)

        return total_reward


if __name__ == "__main__":
    ppo = PPO(env_name="Pendulum-v1")
    ppo.train()
    print(ppo.test(render=True))
