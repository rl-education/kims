"""FrozenLake human play environment."""
import sys
import termios
import tty

import gym

ARROW_KEYS = {
    "\x1b[A": 3,
    "\x1b[B": 1,
    "\x1b[C": 2,
    "\x1b[D": 0,
}


class _Getch:
    def __call__(self) -> str:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
            if ch not in ARROW_KEYS:
                raise InterruptedError
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def main() -> None:
    inkey = _Getch()

    env = gym.make("FrozenLake8x8-v1", is_slippery=False)

    for episode_idx in range(3):
        state = env.reset()

        while True:
            env.render()

            # Choose an action from keyboard
            action = ARROW_KEYS[inkey()]
            next_state, reward, done, _ = env.step(action)

            # Show the board after action
            env.render()

            print(
                f"\nstate: {state}\n"
                f"action: {action}\n"
                f"reward: {reward}\n"
                f"next_state: {next_state}\n"
                f"done: {done}\n",
            )

            state = next_state

            if done:
                print(f"\nepisode_idx: {episode_idx}\n")
                break


if __name__ == "__main__":
    main()
