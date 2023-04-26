"""Policy iteration solution."""
import time
from typing import Any

import gym
import numpy as np
from day_1.ch2_dp.visualization import visualize_results


def policy_evaluation(
    policy_table: np.ndarray,
    dynamics: dict[int, Any],
    gamma: float = 0.99,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Policy evaluation."""
    # Initialize value table
    value_table = np.random.uniform(size=(observ_num, 1))

    # Run while loop until value table converge
    while True:
        value_prime = np.zeros((observ_num, 1))
        for s in dynamics:
            for a in dynamics[s]:
                for trans_prob, next_obs, reward, _ in dynamics[s][a]:
                    value_prime[s][0] += (
                        policy_table[s][a] * trans_prob * (reward + gamma * value_table[next_obs])
                    )

        distance = np.max(np.abs(value_table - value_prime))
        value_table = value_prime

        # If the distance between value table and value prime is not smaller than epsilon, reiterate loop
        if distance < epsilon:
            break
    return value_table


def policy_improvement(value_table: np.ndarray, dynamics: dict[int, Any], gamma: float = 0.99) -> np.ndarray:
    """Policy improvement."""
    # Initialize policy prime and Q-function table
    policy_prime = np.zeros((observ_num, action_num))
    q_table = np.zeros((observ_num, action_num))

    # Update Q-function table through policy improvement
    for s in dynamics:
        for a in dynamics[s]:
            for trans_prob, next_obs, reward, _ in dynamics[s][a]:
                q_table[s][a] += trans_prob * (reward + gamma * value_table[next_obs])

    # Update policy table from the action with highest Q-value as 1 at the current state
    policy_prime[np.arange(observ_num), np.argmax(q_table, axis=1)] = 1
    return policy_prime


def policy_iteration(env: gym.Env, observ_num: int, action_num: int) -> tuple[np.ndarray, np.ndarray]:
    """Policy iteration."""
    dynamics = env.unwrapped.P
    policy_table = np.random.uniform(size=(observ_num, action_num))
    policy_table = policy_table / np.sum(policy_table, axis=1, keepdims=True)

    iterations = 0
    while True:
        value_table = policy_evaluation(policy_table, dynamics)
        policy_prime = policy_improvement(value_table, dynamics)

        # If the policy table is not equal to the policy prime, reiterate loop
        if (policy_table == policy_prime).all():
            break

        iterations += 1
        print(f"iterations: {iterations}")

        # Change the policy table to the policy prime
        policy_table = policy_prime
    return policy_table, value_table


def run(env: gym.Env, action_num: int, policy_table: np.ndarray) -> float:
    """Run episodes."""
    done = False
    total_reward = 0.0
    obs, _ = env.reset()

    while not done:
        env.render()

        action = np.random.choice(action_num, 1, p=policy_table[obs][:])[0]
        next_obs, reward, done, _, _ = env.step(action)

        total_reward += reward
        obs = next_obs
        time.sleep(0.3)
    return total_reward


if __name__ == "__main__":
    env = gym.make("FrozenLake8x8-v1", is_slippery=False, render_mode="human")

    observ_num = env.observation_space.n
    action_num = env.action_space.n
    print(f"observ_num: {observ_num} | action_num: {action_num}")

    start_time = time.time()
    policy_table, value_table = policy_iteration(env=env, observ_num=observ_num, action_num=action_num)
    end_time = time.time() - start_time

    visualize_results(policy=policy_table, value=np.reshape(value_table, (8, 8)), title="Policy Iteration")

    sum_returns = 0.0
    num_episodes = 3
    for _ in range(num_episodes):
        episode_return = run(env=env, action_num=action_num, policy_table=policy_table)
        sum_returns += episode_return

    print(
        f"\ntotal_time: {end_time}\n"
        f"num_episodes: {num_episodes}\n"
        f"sum_returns: {sum_returns}\n"
        f"mean_returns: {sum_returns / num_episodes}\n",
    )
