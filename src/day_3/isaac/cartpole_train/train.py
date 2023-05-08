from omni.isaac.gym.vec_env import VecEnvBase
from .ppo import PPO

# this need to be started before calling the omni... libraries
env = VecEnvBase(headless=True)

from cartpole import CartpoleTask
task = CartpoleTask(name="Cartpole")
env.set_task(task, backend="torch")

# create custom PPO class 
model = PPO(env)
model.train()
env.close()
