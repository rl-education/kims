"""PPO implementation for Pendulum-v1 environment."""

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim


def plot(frame_idx: int, rewards: list[np.ndarray]) -> None:
    """Draw reward graph."""
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title(f"frame {frame_idx}. reward: {rewards[-1]}")
    plt.plot(rewards)
    plt.show()


class ActorCritic(nn.Module):
    """Actor-Critic network."""

    def __init__(self, num_inputs: int, num_outputs: int, hidden_size: int):
        super().__init__()

        self.critic = nn.Sequential(nn.Linear(num_inputs, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) / 10)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        action_distribution = torch.distributions.Normal(mu, std)
        return action_distribution, value


class PPO:
    """PPO method."""

    def __init__(self, envs: gym.Env, model: nn.Module, optimizer: optim.Optimizer):
        self.envs = envs
        self.model = model
        self.optimizer = optimizer

    def compute_gae(
        self,
        next_value: torch.Tensor,
        rewards: list[torch.Tensor],
        masks: list[torch.Tensor],
        values: list[torch.Tensor],
        gamma: float = 0.99,
        tau: float = 0.95,
    ) -> list[torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        values = values + [next_value]
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])

        return returns

    def ppo_iter(
        self,
        mini_batch_size: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO iteration."""
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = torch.randint(0, batch_size, (mini_batch_size,))
            yield (
                states[rand_ids, :],
                actions[rand_ids, :],
                log_probs[rand_ids, :],
                returns[rand_ids, :],
                advantage[rand_ids, :],
            )

    def ppo_update(
        self,
        ppo_epochs: int,
        mini_batch_size: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        clip_param: float = 0.2,
    ) -> None:
        """Update model."""
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(
                mini_batch_size,
                states,
                actions,
                log_probs,
                returns,
                advantages,
            ):
                dist, value = model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def learn(
        self,
        max_frames: int,
        num_steps: int,
        mini_batch_size: int,
        ppo_epochs: int,
    ) -> None:
        """Learn model with PPO."""
        state, _ = envs.reset()

        frame_idx = 0
        test_rewards = []
        while frame_idx < max_frames:
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []
            entropy = 0

            for _ in range(num_steps):
                state = torch.FloatTensor(state)
                dist, value = model(state)

                action = dist.sample()
                next_state, reward, terminate, truncate, _ = self.envs.step(action.cpu().numpy())
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
                    next_state, _ = self.envs.reset()
                state = next_state
                frame_idx += 1

                if frame_idx % 1000 == 0:
                    test_reward = np.mean([self.test() for _ in range(10)])
                    test_rewards.append(test_reward)

            next_state = torch.FloatTensor(next_state)  # type: ignore[reportUnboundVariable]
            _, next_value = model(next_state)
            returns = self.compute_gae(next_value, rewards, masks, values)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns - values

            self.ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)
            print(frame_idx)
        plot(frame_idx, test_rewards)

    def test(self, render: bool = False) -> None:
        """."""
        env = gym.make("Pendulum-v1", render_mode="human") if render else gym.make("Pendulum-v1")
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            dist, _ = model(state)
            next_state, reward, terminate, truncate, _ = env.step(dist.sample().cpu().numpy()[0])
            done = terminate or truncate
            state = next_state
            total_reward += reward

        return total_reward


if __name__ == "__main__":
    # freeze_support()
    envs = gym.vector.make("Pendulum-v1", num_envs=16)
    num_inputs = envs.observation_space.shape[1]
    num_outputs = envs.action_space.shape[1]

    # Hyper params:
    hidden_size = 256
    lr = 3e-4
    max_frames = 15000
    num_steps = 20
    mini_batch_size = 5
    ppo_epochs = 4
    threshold_reward = -200

    model = ActorCritic(num_inputs, num_outputs, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ppo = PPO(envs=envs, model=model, optimizer=optimizer)
    ppo.learn(
        max_frames=max_frames,
        num_steps=num_steps,
        mini_batch_size=mini_batch_size,
        ppo_epochs=ppo_epochs,
    )
    print(ppo.test())
