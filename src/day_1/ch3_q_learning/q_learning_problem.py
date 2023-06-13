"""Q-learning solution."""
import time

import gym
import numpy as np
from visualization import visualize_results


class QLearning:
    def __init__(
        self,
        env: gym.Env,
        state_num: int,
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
        ###
        # Problem 1:
        # Please write the code that choose the action with highest Q-value
        # at the current state with probability 1 - epsilon
        # and a random action with probability epsilon
        if np.random.rand() > self.epsilon:
            action = None
        else:
            action = None
        ###
        return action

    def update_q_table(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> None:
        q_value = self.q_table[state][action]
        ###
        # Problem 2:
        # Please write the code to update the Q-function table using the TD backup
        self.q_table[state][action] += None
        ###

    def train(self) -> np.ndarray:
        start_time = time.time()

        episode_idx = 0
        episode_reward = 0
        episode_returns = 0

        state = self.env.reset()
        for step_idx in range(1, self.num_steps + 1):
            # Collect experience (s, a, r, s') using the policy
            action = self.select_action(state=state)
            next_state, reward, done, _ = self.env.step(action)

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
                    f"episode_returns: {episode_returns}\n",
                )

                state = self.env.reset()
                episode_returns += episode_reward
                episode_idx += 1
                episode_reward = 0

            # If all the returns of episodes are above 1000, stop training
            if episode_returns >= 1000:
                break
        return self.q_table

    def test(self) -> None:
        episode_reward = 0

        for episode_idx in range(3):
            state = self.env.reset()

            while True:
                time.sleep(1)
                self.env.render()

                action = np.argmax(self.q_table[state])
                next_state, reward, done, _ = self.env.step(action)

                state = next_state
                episode_reward += reward

                if done:
                    self.env.render()
                    print(
                        f"\nepisode_idx: {episode_idx}\n" f"episode_reward: {episode_reward}\n",
                    )

                    episode_reward = 0
                    break


def main() -> None:
    env = gym.make("FrozenLake8x8-v1", is_slippery=False)

    state_num = env.observation_space.n
    action_num = env.action_space.n
    print(f"state_num: {state_num} | action_num: {action_num}\n")

    # Create Q-learning agent
    q_learning = QLearning(env=env, state_num=state_num, action_num=action_num)

    # Train Q-learning agent
    q_table = q_learning.train()

    # Visualize Q table
    visualize_results(q_table=q_table)

    # Test Q-learning agent
    q_learning.test()

    # Close the FrozenLake8x8 environment
    env.close()


if __name__ == "__main__":
    main()
