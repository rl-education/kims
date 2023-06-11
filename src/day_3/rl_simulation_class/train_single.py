from pathlib import Path

from rl_simulation_class.utils.config import Config, SB3Config, load_config
from rl_simulation_class.utils.vec_env import VecEnvBaseLivestream
from stable_baselines3 import PPO


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
    from rl_simulation_class.part_1.simple_cartpole import CartpoleTask

    task = CartpoleTask(name="Cartpole", config=config, env=env)
    env.set_task(task, backend="torch")
    # create agent from stable baselines
    assert isinstance(config.policy_config, SB3Config)
    if config.checkpoint:
        path = Path(config.checkpoint)
    else:
        path = Path(config.policy_config.tensorboard_log) / "model"
    if config.test:
        from gym.wrappers.monitor import Monitor
        from stable_baselines3.common.evaluation import evaluate_policy

        env = Monitor(env, path.with_suffix(".mp4"), force=True)
        model = PPO.load(path.with_suffix(".zip"))
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    else:
        model = PPO(
            config.policy_config.policy,  # policy type
            env,
            n_steps=config.policy_config.n_steps,
            batch_size=config.policy_config.batch_size,
            n_epochs=config.policy_config.n_epochs,
            learning_rate=config.policy_config.learning_rate,
            gamma=config.policy_config.gamma,
            device=config.policy_config.device,
            ent_coef=config.policy_config.ent_coef,
            vf_coef=config.policy_config.vf_coef,
            max_grad_norm=config.policy_config.max_grad_norm,
            verbose=config.policy_config.verbose,
            tensorboard_log=config.policy_config.tensorboard_log,
        )

        model.learn(total_timesteps=config.policy_config.max_steps)
        model.save(path)


def main() -> None:
    """Run the training."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config/")
    parser.add_argument("--config_file", type=str, default="config_single.yaml")

    args = parser.parse_args()

    config: Config = load_config(args.config_path, args.config_file)
    env = get_env(config)
    train(config, env)
    env.close()


if __name__ == "__main__":
    main()
