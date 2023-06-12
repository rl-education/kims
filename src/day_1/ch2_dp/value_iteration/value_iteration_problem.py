"""Value iteration solution."""
import time

import gym
import numpy as np
from visualization import visualize_results


def value_iteration(
    env: gym.Env,
    state_num: int,
    action_num: int,
    gamma: float = 0.99,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    # Initialize dynamics
    dynamics = env.unwrapped.P

    # Initialize value table and policy table
    policy_table = np.zeros((state_num, action_num))
    value_table = np.random.uniform(size=(state_num, 1))

    iterations = 0
    while True:
        q_table = np.zeros((state_num, action_num))
        ###
        # Problem 1:
        # Please write the code to update the Q-function table by solving the Bellman optimality equation
        ###

        ###
        # Problem 2:
        # Please write the code to update value prime V' from the highest Q-value at the Q-function table
        value_prime = None
        ###

        distance = np.max(np.abs(value_table - value_prime))
        value_table = value_prime

        # If the distance between value table and value prime is not smaller than epsilon, reiterate loop
        if distance < epsilon:
            break

        iterations += 1
        print(f"iterations: {iterations}")

    # Set to 1 the entry in the policy table that represents taking the action with the highest Q-value at the current state
    policy_table[np.arange(state_num), np.argmax(q_table, axis=1)] = 1
    return policy_table, value_table


def test(env: gym.Env, action_num: int, policy_table: np.ndarray) -> float:
    episode_reward = 0

    for episode_idx in range(3):
        state = env.reset()

        while True:
            time.sleep(1)
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

    # Run value iteration
    start_time = time.time()
    policy_table, value_table = value_iteration(env=env, state_num=state_num, action_num=action_num)
    print(f"\ntotal_time: {time.time() - start_time}\n")

    # Visualize policy and value
    visualize_results(policy=policy_table, value=np.reshape(value_table, (8, 8)))

    # Test PI agent using policy table
    test(env=env, action_num=action_num, policy_table=policy_table)


if __name__ == "__main__":
    main()
