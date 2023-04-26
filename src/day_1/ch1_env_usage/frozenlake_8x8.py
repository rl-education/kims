"""FrozenLake 8x8 environment."""
import time

import gym


def main() -> None:
    """Run FrozenLake 8x8 environment."""
    env = gym.make("FrozenLake8x8-v1", is_slippery=False, render_mode="human")

    observ_num = env.observation_space.n
    action_num = env.action_space.n
    print(f"observ_num: {observ_num} | action_num: {action_num}")

    for _ in range(10000):
        done = False
        obs, _ = env.reset()

        while not done:
            env.render()

            action = env.action_space.sample()
            next_obs, reward, done, truncated, _ = env.step(action)

            print(
                f"observation: {obs} | "
                f"action: {action} | "
                f"reward: {reward} | "
                f"next_observation: {next_obs} | "
                f"done: {done} | "
                f"truncated: {truncated}\n",
            )

            obs = next_obs
            time.sleep(0.5)


if __name__ == "__main__":
    main()
