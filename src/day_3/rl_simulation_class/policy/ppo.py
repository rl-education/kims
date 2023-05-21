"""PPO implementation for Pendulum-v1 environment.

Reference:
    - https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb
"""

from typing import Generator, List, Tuple

import numpy as np
import torch
from gym import Space
from omni.isaac.gym.vec_env import VecEnvBase
from rl_simulation_class.utils.config import Config, PPOConfig
from torch import nn, optim
from torch.nn import functional as F
from tqdm import trange


class ActorNetwork(nn.Module):
    """Actor Network which decide action."""

    def __init__(self, state_dim: int, num_actions: int, hidden_size: int = 256):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

        self.log_std = nn.Parameter(torch.ones(1, num_actions) * 0.0)

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal:
        """Compute action distributions."""
        mu = self.layers(x)
        std = self.log_std.exp().expand_as(mu)
        action_distribution = torch.distributions.Normal(mu, std)
        return action_distribution


class CriticNetwork(nn.Module):
    """Critic Network which predict action value."""

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
        env: VecEnvBase,
        config: Config,
    ) -> None:
        self.env = env
        self.num_envs = self.env.num_envs
        print(self.env)
        print(self.env.observation_space)
        print(self.env.action_space)
        print(self.env.observation_space.shape)
        print(self.env.action_space.shape)
        state_dim = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.shape[0]
        print(state_dim)
        print(num_actions)
        ppo_config = config.policy_config
        self.device = config.device
        assert isinstance(ppo_config, PPOConfig)
        # Create actor-critic network
        self.actor = ActorNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_size=ppo_config.hidden_size,
        ).to(self.device)

        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_size=ppo_config.hidden_size,
        ).to(self.device)

        # Create an optimizer
        self.optimizer = optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": ppo_config.learning_rate},
                {"params": self.critic.parameters(), "lr": ppo_config.learning_rate},
            ],
        )

        # Define hyperparameter for training
        self.num_steps = ppo_config.num_steps
        self.batch_size = ppo_config.batch_size
        self.mini_batch_size = ppo_config.mini_batch_size
        self.ppo_epochs = ppo_config.ppo_epochs
        self.gamma = ppo_config.gamma
        self.tau = ppo_config.tau
        self.clip_param = ppo_config.clip_param

    def compute_gae(
        self,
        next_value: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        values = torch.cat([values, next_value])
        gae = 0
        returns = []

        for step in reversed(range(rewards.shape[0])):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.tau * masks[step] * gae
            returns.insert(0, gae + values[step])

        return torch.stack(returns).detach()

    def ppo_iter(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantage: torch.Tensor,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
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
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                # surr2 *= advantage

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(return_, value)

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def train(
        self,
    ) -> None:
        """Learn model with PPO."""
        state = self.env.reset()

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

            while len(states) * self.num_envs < self.batch_size:
                dist = self.actor(state)
                value = self.critic(state)

                action = dist.sample()
                next_state, reward, done, _ = self.env.step(action)

                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                masks.append((torch.ones(done.shape[0], device=self.device) - done))

                states.append(state)
                actions.append(action)

                state = next_state
                step += 1

            values = torch.cat(values).detach()
            log_probs = torch.cat(log_probs).detach()
            actions = torch.cat(actions)
            states = torch.cat(states)
            rewards = torch.cat(rewards)
            masks = torch.cat(masks)
            next_value = self.critic(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)
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
            print(test_reward)

    def test(self) -> float:
        """."""
        state = self.env.reset()
        done = False
        total_reward = []
        all_done = torch.zeros(self.num_envs, device=self.device)
        while not done:
            dist = self.actor(state)
            action = dist.sample()
            next_state, reward, done_tensor, _ = self.env._task.run_test(action)
            state = next_state
            mask = torch.ones(done_tensor.shape[0], device=self.device) - done_tensor
            reward = reward * mask
            total_reward.append(reward)
            all_done += done_tensor
            done = all_done.all()
        return torch.stack(total_reward).sum(dim=1).mean().item()
