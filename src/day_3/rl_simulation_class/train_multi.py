import dataclasses

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_simulation_class.utils.config import Config, RLGamesConfig, load_config
from rl_simulation_class.utils.vec_env import VecEnvBaseLivestream

config: Config = load_config(
    "/home/yuri/personal_workspace/isaac/kims/src/day_3/config/",
    "config_multi.yaml",
)
# this need to be started before calling the omni... libraries
env = VecEnvBaseLivestream(launch_type=config.launch_type)

from rl_simulation_class.part_2.multi_cartpole import CartpoleTask
from rl_simulation_class.part_2.multi_env_rl_task import RLGPUAlgoObserver, RLGPUEnv


class RLGTrainer:
    def __init__(self, env, config):
        # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.

        # register the rl-games adapter to use inside the runner
        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "rlgpu",
            {
                "vecenv_type": "RLGPU",
                "env_creator": lambda **kwargs: env,
            },
        )
        self.rlg_config_dict = dataclasses.asdict(config)

    def run(self):
        # create runner and set the settings
        runner = Runner(RLGPUAlgoObserver())
        runner.load(self.rlg_config_dict)
        runner.reset()

        runner.run(
            {
                "train": True,
                "play": False,
                "checkpoint": None,
                "sigma": None,
            },
        )


task = CartpoleTask(name="Cartpole", config=config, env=env)
env.set_task(task, backend="torch")

assert isinstance(config.policy_config, RLGamesConfig)
rlg_trainer = RLGTrainer(env, config.policy_config)
rlg_trainer.run()
env.close()
