"""CartPole environment."""

import time

import gym


def main() -> None:
    env = gym.make("CartPole-v1")

    observ_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    print(f"observ_dim: {observ_dim} | action_num: {action_num}")

    for _ in range(10000):
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
            time.sleep(0.3)


if __name__ == "__main__":
    main()
