from omni.isaac.gym.vec_env import VecEnvBase
from rl_simulation_class.ppo import PPO

# this need to be started before calling the omni... libraries
env = VecEnvBase(headless=False, enable_viewport=True)
# env = VecEnvBase(headless=True)

# from omni.isaac.core.utils.extensions import enable_extension
# enable_extension("omni.kit.viewport.bundle")
# enable_extension("omni.replicator.isaac")
# from rl_simulation_class.utils.config import load_config, Config
# from rl_simulation_class.tasks.simple_cartpole import CartpoleTask

# config: Config = load_config("/home/yuri/personal-workspace/kims/src/day_3/config", "config.yaml")

# task = CartpoleTask(name="Cartpole", config=config, env=env)
from rl_simulation_class.cartpole import CartpoleTask

task = CartpoleTask(name="Cartpole")
env.set_task(task, backend="torch")

# create custom PPO class
model = PPO(env)
model.train()
env.close()
