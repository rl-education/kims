"""Q-learning solution."""
import time

import gym
import numpy as np


def select_action(obs: int, action_num: int, q_table: np.ndarray, epsilon: float = 0.8) -> np.ndarray:
    """Select action."""
    if np.random.rand() <= epsilon:
        # Choose a random action with probability epsilon
        return np.random.randint(action_num)
    else:
        # Choose the action with highest Q-value at the current state
        return np.argmax(q_table[obs])


def update_q_table(
    obs: int,
    action: int,
    reward: float,
    next_obs: int,
    q_table: np.ndarray,
    gamma: float = 0.99,
    learning_rate: float = 0.01,
) -> None:
    """Update Q-function table."""
    q = q_table[obs][action]
    q_backup = reward + gamma * max(q_table[next_obs])
    q_table[obs][action] += learning_rate * (q_backup - q)
    return q_table


def run(
    env: gym.Env,
    action_num: int,
    q_table: np.ndarray,
    test_mode: bool = False,
) -> tuple[float, np.ndarray]:
    """Run episodes."""
    done = False
    total_reward = 0.0
    obs, _ = env.reset()

    while not done:
        if test_mode:
            action = np.argmax(q_table[obs])
            next_obs, reward, done, _, _ = env.step(action)
        else:
            action = select_action(obs=obs, action_num=action_num, q_table=q_table)
            next_obs, reward, done, _, _ = env.step(action)
            q_table = update_q_table(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                q_table=q_table,
            )

        total_reward += reward
        obs = next_obs
    return total_reward, q_table


def train(env: gym.Env, action_num: int, q_table: np.ndarray) -> np.ndarray:
    """Train Q-learning agent."""
    start_time = time.time()

    sum_returns = 0.0
    num_episodes = 0
    for _ in range(100000):
        episode_return, q_table = run(env=env, action_num=action_num, q_table=q_table)
        sum_returns += episode_return
        num_episodes += 1

        print(
            f"\ntotal_time: {time.time() - start_time}\n"
            f"num_episodes: {num_episodes}\n"
            f"sum_returns: {sum_returns}\n"
            f"mean_returns: {sum_returns / num_episodes}\n",
        )
    return q_table


def test(env: gym.Env, action_num: int, q_table: np.ndarray) -> None:
    """Test Q-learning agent."""
    sum_returns = 0.0
    num_episodes = 0
    for _ in range(3):
        episode_return, _ = run(env=env, action_num=action_num, q_table=q_table, test_mode=True)
        sum_returns += episode_return
        num_episodes += 1

        print(
            f"num_episodes: {num_episodes}\n"
            f"sum_returns: {sum_returns}\n"
            f"mean_returns: {sum_returns / num_episodes}\n",
        )


if __name__ == "__main__":
    env = gym.make("FrozenLake8x8-v1", is_slippery=False)

    observ_num = env.observation_space.n
    action_num = env.action_space.n
    print(f"observ_num: {observ_num} | action_num: {action_num}")

    # Initialize Q-function table
    q_table = np.zeros((observ_num, action_num))

    # Train Q-learning agent
    q_table = train(env=env, action_num=action_num, q_table=q_table)

    # Test Q-learning agent
    env = gym.make("FrozenLake8x8-v1", is_slippery=False, render_mode="human")
    test(env=env, action_num=action_num, q_table=q_table)
