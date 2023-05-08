"""FrozenLake 4x4 environment."""
import time

import gym


def main() -> None:
    """Run FrozenLake 4x4 environment."""
    env = gym.make("FrozenLake-v1", is_slippery=False)

    observ_num = env.observation_space.n
    action_num = env.action_space.n
    print(f"observ_num: {observ_num} | action_num: {action_num}")

    done = False
    obs = env.reset()

    while not done:
        env.render()

        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)

        print(
            f"observation: {obs} | "
            f"action: {action} | "
            f"reward: {reward} | "
            f"next_observation: {next_obs} | "
            f"done: {done}\n",
        )

        obs = next_obs
        time.sleep(1)


if __name__ == "__main__":
    main()
