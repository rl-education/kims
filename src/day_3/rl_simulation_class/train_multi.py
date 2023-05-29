from rl_simulation_class.utils.config import Config, RLGamesConfig, load_config
from rl_simulation_class.utils.vec_env import VecEnvBaseLivestream

config: Config = load_config(
    "/home/yuri/personal_workspace/isaac/kims/src/day_3/config/",
    "config_multi.yaml",
)
# this need to be started before calling the omni... libraries
env = VecEnvBaseLivestream(launch_type=config.launch_type)

from rl_simulation_class.part_2.multi_cartpole import CartpoleTask
from rl_simulation_class.part_2.multi_env_rl_task import RLGTrainer

task = CartpoleTask(name="Cartpole", config=config, env=env)
env.set_task(task, backend="torch")

assert isinstance(config.policy_config, RLGamesConfig)
rlg_trainer = RLGTrainer(env, config.policy_config)
rlg_trainer.run(not config.test, config.checkpoint)
env.close()

if __name__ == "__main__":
    pass
