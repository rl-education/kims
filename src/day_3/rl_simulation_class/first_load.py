import glob
import time

import requests


def wait_for_isaac_to_load():
    """
    Wait until the log string is found in the log file.
    """
    isaac_loaded_string = "Isaac Sim Headless Native App is loaded."
    log_gpath = "/root/.nvidia-omniverse/logs/Kit/Isaac-Sim/*/*.log"

    files = []
    while not files:
        files = glob.glob(log_gpath)
        time.sleep(1)
    log_file = files[0]
    while isaac_loaded_string not in open(log_file).read():
        time.sleep(1)


def main() -> None:
    """Initialize IsaacSim Once to load the RTX shaders"""
    print("Waiting for IsaacSim...")
    print("This process can take around 5-10 minutes the first time")
    wait_for_isaac_to_load()
    print("First load successful!")
    print(f"The Instance IP is: {requests.get('https://api.ipify.org').text}")


if __name__ == "__main__":
    main()
