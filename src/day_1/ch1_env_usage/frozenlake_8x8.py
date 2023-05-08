"""FrozenLake 8x8 environment."""
import time

import gym


def main() -> None:
    env = gym.make("FrozenLake8x8-v1", is_slippery=False)

    state_num = env.observation_space.n
    action_num = env.action_space.n
    print(f"state_num: {state_num} | action_num: {action_num}\n")

    for episode_idx in range(3):
        state = env.reset()

        while True:
            time.sleep(1)
            env.render()

            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            print(
                f"state: {state} | "
                f"action: {action} | "
                f"reward: {reward} | "
                f"next_state: {next_state} | "
                f"done: {done}\n",
            )

            state = next_state

            if done:
                env.render()
                print(f"\nepisode_idx: {episode_idx}\n")
                break


if __name__ == "__main__":
    main()
