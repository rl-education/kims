"""Q-learning solution."""
import time

import gym
import numpy as np
from visualization import visualize_results


class QLearning:
    def __init__(
        self,
        env: gym.Env,
        action_num: int,
        gamma: float = 0.99,
        learning_rate: float = 0.01,
        num_steps: int = 10000000,
        epsilon: float = 0.8,
    ) -> None:
        self.env = env
        self.action_num = action_num
        self.gamma = gamma
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # Initialize Q-function table
        self.q_table = np.zeros((state_num, action_num))

    def select_action(self, state: int) -> np.ndarray:
        # Choose the action with highest Q-value at the current state
        if np.random.rand() > self.epsilon:
            action = np.argmax(self.q_table[state])
        # Choose a random action with probability epsilon
        else:
            action = np.random.randint(self.action_num)
        return action

    def update_q_table(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> None:
        q_value = self.q_table[state][action]
        expected_q_value = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (expected_q_value - q_value)

    def train(self) -> np.ndarray:
        start_time = time.time()

        episode_idx = 0
        episode_reward = 0
        episode_returns = 0.0

        state = self.env.reset()
        for step_idx in range(1, self.num_steps + 1):
            # Collect experience (s, a, r, s') using the policy
            action = self.select_action(state=state)
            next_state, reward, done, _ = env.step(action)

            # Update Q table
            self.update_q_table(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
            )

            state = next_state
            episode_reward += reward

            if done:
                print(
                    f"\ntotal_time: {time.time() - start_time}\n"
                    f"step_idx: {step_idx}\n"
                    f"episode_idx: {episode_idx}\n"
                    f"episode_reward: {episode_reward}\n"
                    f"episode_returns: {episode_returns}\n",
                )

                state = self.env.reset()
                episode_returns += episode_reward
                episode_idx += 1
                episode_reward = 0

            # If all the returns of episodes are above 1000, stop training
            if episode_returns >= 1000.0:
                break
        return self.q_table

    def test(self) -> None:
        episode_idx = 0
        episode_reward = 0

        state = self.env.reset()
        for step_idx in range(42):
            time.sleep(0.5)
            self.env.render()

            action = np.argmax(q_table[state])
            next_state, reward, done, _ = self.env.step(action)

            state = next_state
            episode_reward += reward

            if done:
                print(
                    f"\nstep_idx: {step_idx}\n"
                    f"episode_idx: {episode_idx}\n"
                    f"episode_reward: {episode_reward}\n",
                )

                state = self.env.reset()
                episode_idx += 1
                episode_reward = 0


if __name__ == "__main__":
    env = gym.make("FrozenLake8x8-v1", is_slippery=False)

    state_num = env.observation_space.n
    action_num = env.action_space.n
    print(f"state_num: {state_num} | action_num: {action_num}\n")

    # Create Q-learning agent
    q_learning = QLearning(env=env, action_num=action_num)

    # Train Q-learning agent
    q_table = q_learning.train()

    # Visualize Q table
    visualize_results(q_table=q_table)

    # Test Q-learning agent
    q_learning.test()

    # Close the FrozenLake8x8 environment
    env.close()
