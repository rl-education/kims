from rl_simulation_class.utils.config import Config, SB3Config, load_config
from rl_simulation_class.utils.vec_env import VecEnvBaseLivestream
from stable_baselines3 import PPO

config: Config = load_config(
    "/home/yuri/personal_workspace/isaac/kims/src/day_3/config/",
    "config_single.yaml",
)
# this need to be started before calling the omni... libraries
env = VecEnvBaseLivestream(launch_type=config.launch_type)

from rl_simulation_class.part_1.simple_cartpole import CartpoleTask

task = CartpoleTask(name="Cartpole", config=config, env=env)
env.set_task(task, backend="torch")
assert isinstance(config.policy_config, SB3Config)
# create agent from stable baselines
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

model.learn(total_timesteps=100000)
model.save("ppo_cartpole")

env.close()
