"""CartPole environment."""

import time

import gym


def main() -> None:
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    print(f"state_dim: {state_dim} | action_num: {action_num}\n")

    state = env.reset()
    for step_idx in range(10000):
        time.sleep(0.3)
        env.render()

        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)

        print(
            f"\nstate: {state}\n"
            f"action: {action}\n"
            f"reward: {reward}\n"
            f"next_state: {next_state}\n"
            f"done: {done}\n",
        )

        state = next_state

        if done:
            print(f"step_idx: {step_idx + 1}\n")
            state = env.reset()


if __name__ == "__main__":
    main()
