import time

import requests
from rl_simulation_class.utils.vec_env import IsaacLaunchType, VecEnvBaseLivestream


def main() -> None:
    """Initialize IsaacSim with Websocket Streaming"""
    print("Starting for IsaacSim...")
    env = VecEnvBaseLivestream(launch_type=IsaacLaunchType.WEBSOCKET)
    print(f"The Instance IP is: {requests.get('https://api.ipify.org').text}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping IsaacSim...")
        env.close()


if __name__ == "__main__":
    main()
