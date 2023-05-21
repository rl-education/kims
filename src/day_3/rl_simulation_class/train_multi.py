from omni.isaac.gym.vec_env import VecEnvBase
from rl_simulation_class.policy.ppo import PPO

# this need to be started before calling the omni... libraries
env = VecEnvBase(headless=False, enable_viewport=True)
# env = VecEnvBase(headless=True)

from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.kit.viewport.bundle")
enable_extension("omni.replicator.isaac")
from rl_simulation_class.tasks.multi_cartpole import CartpoleTask
from rl_simulation_class.utils.config import Config, load_config

config: Config = load_config("/home/yuri/personal-workspace/kims/src/day_3/config", "config.yaml")

task = CartpoleTask(name="Cartpole", config=config, env=env)
env.set_task(task, backend="torch")

# create custom PPO class
model = PPO(env, config)
model.train()
env.close()
