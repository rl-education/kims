import time

import requests
from rl_simulation_class.utils.config import Config
from rl_simulation_class.utils.vec_env import IsaacLaunchType, VecEnvBaseLivestream


def get_task(env, config):
    """Get an empty task"""
    from rl_simulation_class.part_1.simple_rl_task import SimpleRLTask

    class EmptyTask(SimpleRLTask):
        """EmptyTask class for VecEnvBaseLivestream"""

        def reset(self):
            pass

        def get_observations(self):
            return None

        def calculate_metrics(self):
            return None

        def is_done(self):
            return False

        def pre_physics_step(self, _):
            pass

    return EmptyTask(name="websocket", env=env, config=config)


def main() -> None:
    """Initialize IsaacSim with Websocket Streaming"""
    print("Starting for IsaacSim...")
    env = VecEnvBaseLivestream(launch_type=IsaacLaunchType.WEBSOCKET)
    config = Config(task_config_file="", policy_config_file="", device="cpu", test=True)
    task = get_task(env, config)
    env.set_task(task)
    print(f"The Instance IP is: {requests.get('https://api.ipify.org').text}")
    try:
        while True:
            env.step([])
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping IsaacSim...")
        env.close()


if __name__ == "__main__":
    main()
