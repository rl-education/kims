"""
FrozenLake human play environment.

Note: Only run on Windows OS
"""
import msvcrt

import gym

ARROW_KEYS = {
    b"H": 3,
    b"P": 1,
    b"M": 2,
    b"K": 0,
}


class _Getch:
    def __call__(self) -> str:
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch == b"\xe0":  # indicates an arrow key
                    ch = msvcrt.getch()  # read the next byte
                    if ch not in ARROW_KEYS:
                        raise InterruptedError
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
