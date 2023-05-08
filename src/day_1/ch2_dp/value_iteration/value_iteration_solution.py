"""Value iteration solution."""
import time

import gym
import numpy as np
from visualization import visualize_results


def value_iteration(
    env: gym.Env,
    observ_num: int,
    action_num: int,
    gamma: float = 0.99,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    # Initialize dynamics
    dynamics = env.unwrapped.P

    # Initialize value table and policy table
    policy_table = np.zeros((observ_num, action_num))
    value_table = np.random.uniform(size=(observ_num, 1))

    iterations = 0
    while True:
        q_table = np.zeros((observ_num, action_num))
        for s in dynamics:
            for a in dynamics[s]:
                for trans_prob, next_obs, reward, _ in dynamics[s][a]:
                    q_table[s][a] += trans_prob * (reward + gamma * value_table[next_obs])

        # Update value prime from the highest Q-value at the Q-function table
        value_prime = np.max(q_table, axis=1)

        distance = np.max(np.abs(value_table - value_prime))
        value_table = value_prime

        # If the distance between value table and value prime is not smaller than epsilon, reiterate loop
        if distance < epsilon:
            break

        iterations += 1
        print(f"iterations: {iterations}")

    # Update policy table from the action with highest Q-value as 1 at the current state
    policy_table[np.arange(observ_num), np.argmax(q_table, axis=1)] = 1
    return policy_table, value_table


def run(env: gym.Env, action_num: int, policy_table: np.ndarray) -> float:
    episode_reward = 0

    obs = env.reset()
    for _ in range(100):
        time.sleep(0.5)
        env.render()

        action = np.random.choice(action_num, 1, p=policy_table[obs][:])[0]
        next_obs, reward, done, _ = env.step(action)

        obs = next_obs
        episode_reward += reward

        if done:
            break
    return episode_reward


if __name__ == "__main__":
    env = gym.make("FrozenLake8x8-v1", is_slippery=False)

    observ_num = env.observation_space.n
    action_num = env.action_space.n
    print(f"observ_num: {observ_num} | action_num: {action_num}\n")

    start_time = time.time()
    policy_table, value_table = value_iteration(env=env, observ_num=observ_num, action_num=action_num)
    end_time = time.time() - start_time

    visualize_results(policy=policy_table, value=np.reshape(value_table, (8, 8)))

    num_episodes = 3
    episode_returns = 0.0
    for _ in range(num_episodes):
        episode_reward = run(env=env, action_num=action_num, policy_table=policy_table)
        episode_returns += episode_reward

    print(
        f"\ntotal_time: {end_time}\n"
        f"num_episodes: {num_episodes}\n"
        f"episode_returns: {episode_returns}\n"
        f"mean_returns: {episode_returns / num_episodes}\n",
    )
