import requests
from omni.isaac.gym.vec_env import VecEnvBase


def main() -> None:
    """Initialize IsaacSim Once to load the RTX shaders"""
    print("Loading IsaacSim...")
    print("This process can take around 5-10 minutes the first time")
    env = VecEnvBase(headless=True)
    env.close()
    print("First load successful!")
    print(f"The Instance IP is: {requests.get('https://api.ipify.org').text}")


if __name__ == "__main__":
    main()
