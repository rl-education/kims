"""
DQN algorithm.

Reference:
    - https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb
    - https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb
"""
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import Env
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_num: int) -> None:
        super(QNetwork, self).__init__()

        self.layers = nn.Sequential(
            ###
            # Problem 1:
            # Please write the code that constitutes the layer of the Q-network
            # - Linear transformation (state_dim, 128)
            # - ReLu activation function
            # - Linear transformation (128, 128)
            # - ReLu activation function
            # - Linear transformation (128, action_num)
            ###
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DQN:
    def __init__(
        self,
        env: Env,
        state_dim: int,
        action_num: int,
        gamma: float = 0.99,
        num_steps: int = 1000000,
        batch_size: int = 32,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        use_ddqn: bool = False,
    ) -> None:
        self.env = env
        self.state_dim = state_dim
        self.action_num = action_num
        self.gamma = gamma
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.use_ddqn = use_ddqn

        # Create Q-networks
        self.current_model = QNetwork(state_dim=state_dim, action_num=action_num).to(DEVICE)
        self.target_model = QNetwork(state_dim=state_dim, action_num=action_num).to(DEVICE)

        # Create an optimizer
        self.optimizer = optim.Adam(self.current_model.parameters())

        # Create a replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=1000)

        # Initialize parameters of target model to match parameters of current model
        self.update_target()

        # Initialize tensorbaord to visualize results
        algorithm = "ddqn" if use_ddqn else "dqn"
        self.writer = SummaryWriter(
            log_dir=f"runs/cartpole-{algorithm}-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the set of available actions."""
        ###
        # Problem 2:
        # Please write the code that computes the epsilon value using epsilon greedy exploration
        self.epsilon *= None
        self.epsilon = None
        ###

        # Choose the action with highest Q-value at the current state
        if np.random.rand() > self.epsilon:
            ###
            # Problem 3:
            # Please write the code that obtains the action using Q-value,
            # which is the output of the current Q-network
            q_value = None
            action = None
            ###
        # Choose a random action with probability epsilon
        else:
            action = np.random.randint(self.action_num)
        return action

    def update_target(self) -> None:
        self.target_model.load_state_dict(self.current_model.state_dict())

    def compute_td_loss(self) -> float:
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        action = torch.LongTensor(action).to(DEVICE)
        reward = torch.FloatTensor(reward).to(DEVICE)
        done = torch.FloatTensor(done).to(DEVICE)

        # Predict Q(s)
        q_values = self.current_model(state)
        q_value = q_values.gather(1, action).squeeze(1)

        ###
        # Problem 4:
        # Please write the code to get the target Q-value for Q regression from DQN or Double DQN
        target_q_values = self.target_model(next_state)
        if not self.use_ddqn:  # DQN
            target_q_value = None
        else:  # Double DQN
            next_q_values = None
            target_q_value = None
        ###

        ###
        # Problem 5:
        # Please write the code that obtains the loss function using TD backup and updates parameters of the current model
        expected_q_value = None
        loss = None
        ###

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self) -> None:
        start_time = time.time()

        loss = 0.0
        episode_idx = 0
        episode_reward = 0
        recent_rewards = deque(maxlen=10)

        state = self.env.reset()
        for step_idx in range(1, self.num_steps + 1):
            # Collect experience (s, a, r, s') using the policy
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            # Push experience to replay buffer
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                print(
                    f"\ntotal_time: {time.time() - start_time}\n"
                    f"step_idx: {step_idx}\n"
                    f"episode_idx: {episode_idx}\n"
                    f"episode_reward: {episode_reward}\n"
                    f"loss: {loss}\n",
                )

                # Visualize results within tensorboard
                self.writer.add_scalar("train/episode_reward", episode_reward, episode_idx)
                self.writer.add_scalar("train/loss", loss, episode_idx)

                state = self.env.reset()
                recent_rewards.append(episode_reward)
                episode_idx += 1
                episode_reward = 0

            # Start training when the number of experience is greater than batch_size
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_td_loss()

            # Synchronize parameters of target model as parameters of current model every 100 steps
            if step_idx % 100 == 0:
                self.update_target()

            # If the rewards for all 10 of the most recent episodes are 490, stop training
            if np.mean(recent_rewards) > 490:
                print(f"\nrecent_rewards: {recent_rewards}\n")
                break

    def test(self) -> None:
        episode_idx = 0
        episode_reward = 0

        state = self.env.reset()
        for step_idx in range(5000):
            self.env.render()

            q_value = self.current_model(torch.FloatTensor(np.float32(state)).to(DEVICE))
            action = q_value.argmax().item()
            next_state, reward, done, _ = self.env.step(action)

            state = next_state
            episode_reward += reward

            if done:
                print(
                    f"\nstep_idx: {step_idx + 1}\n"
                    f"episode_idx: {episode_idx}\n"
                    f"episode_reward: {episode_reward}\n",
                )

                self.writer.add_scalar("test/episode_reward", episode_reward, episode_idx)

                state = self.env.reset()
                episode_idx += 1
                episode_reward = 0
