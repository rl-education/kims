"""Policy iteration solution."""
import time
from typing import Any

import gym
import numpy as np
from visualization import visualize_results


def policy_evaluation(
    state_num: int,
    policy_table: np.ndarray,
    dynamics: dict[int, Any],
    gamma: float = 0.99,
    epsilon: float = 1e-6,
) -> np.ndarray:
    # Initialize value table
    value_table = np.random.uniform(size=(state_num, 1))

    # Run while loop until value table converge
    while True:
        value_prime = np.zeros((state_num, 1))
        for s in dynamics:
            for a in dynamics[s]:
                for trans_prob, next_obs, reward, _ in dynamics[s][a]:
                    ###
                    # Problem 1:
                    # Please write the code that computes the value prime V' of the policy π_k
                    # by solving the Bellman expectation equation
                    value_prime[s][0] += None
                    ###

        distance = np.max(np.abs(value_table - value_prime))
        value_table = value_prime

        # If the distance between value table and value prime is not smaller than epsilon, reiterate loop
        if distance < epsilon:
            break
    return value_table


def policy_improvement(
    state_num: int,
    action_num: int,
    value_table: np.ndarray,
    dynamics: dict[int, Any],
    gamma: float = 0.99,
) -> np.ndarray:
    # Initialize policy prime and Q-function table
    policy_prime = np.zeros((state_num, action_num))
    q_table = np.zeros((state_num, action_num))

    # Update Q-function table through policy improvement
    for s in dynamics:
        for a in dynamics[s]:
            for trans_prob, next_obs, reward, _ in dynamics[s][a]:
                ###
                # Problem 2:
                # Please write the code to update the policy to π_{k+1}
                q_table[s][a] += None
                ###

    ###
    # Problem 3:
    # Please write the code to set to 1 the entry in the policy table that represents taking the action with the highest Q-value at the current state
    policy_prime[np.arange(state_num), None] = 1
    ###
    return policy_prime


def policy_iteration(env: gym.Env, state_num: int, action_num: int) -> tuple[np.ndarray, np.ndarray]:
    dynamics = env.unwrapped.P
    policy_table = np.random.uniform(size=(state_num, action_num))
    policy_table = policy_table / np.sum(policy_table, axis=1, keepdims=True)

    iterations = 0
    while True:
        value_table = policy_evaluation(state_num, policy_table, dynamics)
        policy_prime = policy_improvement(state_num, action_num, value_table, dynamics)

        # If the policy table is not equal to the policy prime, reiterate loop
        if (policy_table == policy_prime).all():
            break

        iterations += 1
        print(f"iterations: {iterations}")

        # Change the policy table to the policy prime
        policy_table = policy_prime
    return policy_table, value_table


def test(env: gym.Env, action_num: int, policy_table: np.ndarray) -> None:
    episode_reward = 0

    for episode_idx in range(3):
        state = env.reset()

        while True:
            time.sleep(0.5)
            env.render()

            action = np.random.choice(action_num, 1, p=policy_table[state][:])[0]
            next_state, reward, done, _ = env.step(action)

            state = next_state
            episode_reward += reward

            if done:
                env.render()
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

    # Run policy iteration
    start_time = time.time()
    policy_table, value_table = policy_iteration(env=env, state_num=state_num, action_num=action_num)
    print(f"\ntotal_time: {time.time() - start_time}\n")

    # Visualize policy and value
    visualize_results(policy=policy_table, value=np.reshape(value_table, (8, 8)))

    # Test PI agent using policy table
    test(env=env, action_num=action_num, policy_table=policy_table)


if __name__ == "__main__":
    main()
