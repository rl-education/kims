"""Run DQN."""

import gym
from dqn import DQN


def main() -> None:
    # Initialize a CartPole environment
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    print(f"state_dim: {state_dim} | action_num: {action_num}\n")

    # Create DQN agent
    model = DQN(env=env, state_dim=state_dim, action_num=action_num, use_ddqn=False)

    # Train DQN agent
    model.train()

    # Test DQN agent
    model.test()

    # Close the CartPole environment
    env.close()


if __name__ == "__main__":
    main()
