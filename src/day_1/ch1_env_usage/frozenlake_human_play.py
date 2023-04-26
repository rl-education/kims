"""FrozenLake human play environment."""
import sys
import termios
import tty

import gym


class _Getch:
    def __call__(self) -> str:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


inkey = _Getch()

arrow_keys = {
    "\x1b[A": 3,
    "\x1b[B": 1,
    "\x1b[C": 2,
    "\x1b[D": 0,
}

env = gym.make("FrozenLake8x8-v1", is_slippery=False, render_mode="human")

for _ in range(3):
    done = False
    obs, _ = env.reset()

    while not done:
        # Choose an action from keyboard
        key = inkey()
        if key not in arrow_keys.keys():
            print("Game aborted!")
            break

        action = arrow_keys[key]
        next_obs, reward, done, truncated, _ = env.step(action)

        # Show the board after action
        env.render()

        print(
            f"observation: {obs} | "
            f"action: {action} | "
            f"reward: {reward} | "
            f"next_observation: {next_obs} | "
            f"done: {done} | "
            f"truncated: {truncated}\n",
        )

        obs = next_obs
