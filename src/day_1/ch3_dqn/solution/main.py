"""Run DQN."""

from argparse import ArgumentParser

import gym
from dqn import DQN


def main(use_ddqn: bool) -> None:
    # Initialize a CartPole environment
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    print(f"state_dim: {state_dim} | action_num: {action_num}\n")

    # Create DQN agent
    dqn = DQN(env=env, state_dim=state_dim, action_num=action_num, use_ddqn=use_ddqn)

    # Train DQN agent
    dqn.train()

    # Test DQN agent
    dqn.test()

    # Close the CartPole environment
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use-ddqn", dest="use_ddqn", action="store_true")
    args = parser.parse_args()

    main(use_ddqn=args.use_ddqn)
