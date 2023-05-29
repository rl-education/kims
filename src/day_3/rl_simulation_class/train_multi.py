from rl_simulation_class.utils.config import Config, RLGamesConfig, load_config
from rl_simulation_class.utils.vec_env import VecEnvBaseLivestream


def get_env(config: Config) -> VecEnvBaseLivestream:
    """Creates the IsaacSim environment.

    Args:
        config (Config): The experiment configuration

    Returns:
        VecEnvBaseLivestream: The IsaacSim environment
    """
    # this need to be started before including the omni... libraries
    return VecEnvBaseLivestream(launch_type=config.launch_type)


def train(config: Config, env: VecEnvBaseLivestream) -> None:
    """Train the agent.

    Args:
        config (Config): The experiment configuration
        env (VecEnvBaseLivestream): The IsaacSim environment
    """
    from rl_simulation_class.part_2.multi_cartpole import CartpoleTask
    from rl_simulation_class.part_2.multi_env_rl_task import RLGTrainer

    task = CartpoleTask(name="Cartpole", config=config, env=env)
    env.set_task(task, backend="torch")

    assert isinstance(config.policy_config, RLGamesConfig)
    rlg_trainer = RLGTrainer(env, config.policy_config)
    rlg_trainer.run(not config.test, config.checkpoint)


def main() -> None:
    """Run the training."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/")
    parser.add_argument("--config_file", type=str, default="config_multi.yaml")

    args = parser.parse_args()

    config: Config = load_config(args.config_path, args.config_file)
    env = get_env(config)
    train(config, env)
    env.close()


if __name__ == "__main__":
    main()
