from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from gym import Space, spaces
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.gym.vec_env import VecEnvBase
from rl_simulation_class.utils.config import Config


class SimpleRLTask(BaseTask, ABC):
    """
    Simple Interface for RL tasks.
    """

    def __init__(
        self,
        name: str,
        env: VecEnvBase,
        config: Config,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        """Base Interface for RL tasks.

        Args:
            name (str): name of the task. Needs to be unique.
            env (VecEnvBase): the RL environment
            config (Config): the task configuration
            offset (np.Optional[ndarray], optional): offset applied to all assets of the task. Defaults to None.
        """

        super().__init__(name=name, offset=offset)

        self._config = config
        self._device = self._config.device
        self.test = self._config.test
        print("Task Device:", self._device)

        self._env = env

        # initialize params in case they were not yet initialized by the derived class
        if not hasattr(self, "_num_envs"):
            self._num_envs = 1
        if not hasattr(self, "_num_actions"):
            self._num_actions = 1
        if not hasattr(self, "_num_observations"):
            self._num_observations = 1
        if not hasattr(self, "_num_frames"):
            self._num_frames = 1
        if not hasattr(self, "_num_states"):
            self.num_states = 0
        if not hasattr(self, "action_space"):
            self._action_space = spaces.Box(
                np.ones(self._num_actions) * -1.0,
                np.ones(self._num_actions) * 1.0,
            )
        if not hasattr(self, "observation_space"):
            self._observation_space = spaces.Box(
                np.ones(self._num_observations) * -np.Inf,
                np.ones(self._num_observations) * np.Inf,
            )

        self.cleanup()

    def cleanup(self) -> None:
        """Called before calling a reset() on the world. Reset data structures."""
        self._observations_buffer = torch.zeros(
            (self._num_frames, self.num_observations),
            device=self._device,
            dtype=torch.float,
        )
        self._done_buffer = torch.ones(1, device=self._device, dtype=torch.long)
        self._episodes_count = torch.zeros(1, device=self._device, dtype=torch.long)

    def set_up_scene(self, scene) -> None:
        """Clones environments based on value provided in task config and applies collision filters to mask
            collisions across environments.

        Args:
            scene (Scene): Scene to add objects to.
        """
        super().set_up_scene(scene)

        self._ground_plane_path = "/World/defaultGroundPlane"
        scene.add_default_ground_plane(prim_path=self._ground_plane_path)

        self.set_initial_camera_params(camera_position=[10, 10, 3], camera_target=[0, 0, 0])

    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        """Sets initial camera parameters.

        Args:
            camera_position (list, optional): The camera position. Defaults to [10, 10, 3].
            camera_target (list, optional): The camera target. Defaults to [0, 0, 0].
        """
        set_camera_view(eye=camera_position, target=camera_target, camera_prim_path="/OmniverseKit_Persp")

    @property
    def default_base_path(self):
        """Retrieves default path to the parent of all env prims.

        Returns:
            default_base_path(str): Defaults to "/World/envs".
        """
        return "/World/envs"

    @property
    def num_envs(self):
        """Retrieves number of environments for task.

        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs

    @property
    def num_actions(self):
        """Retrieves dimension of actions.

        Returns:
            num_actions(int): Dimension of actions.
        """
        return self._num_actions

    @property
    def num_observations(self):
        """Retrieves dimension of observations.

        Returns:
            num_observations(int): Dimension of observations.
        """
        return self._num_observations

    @property
    def action_space(self) -> Space:
        """Retrieves action space.

        Returns:
            action_space(gym.Space): Action space.
        """
        return self._action_space

    @property
    def observation_space(self) -> Space:
        """Retrieves observation space.

        Returns:
            observation_space(gym.Space): Observation space.
        """
        return self._observation_space

    ### Methods called by the VecEnvBase
    @abstractmethod
    def reset(self):
        """Reset all environments and return initial observations."""
        raise NotImplementedError

    @abstractmethod
    def pre_physics_step(self, actions):
        """Optionally implemented by individual task classes to process actions.

        Args:
            actions (torch.Tensor): Actions generated by RL policy.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self):
        """
        Return the observations
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_metrics(self):
        """
        Return the rewards
        """
        raise NotImplementedError

    @abstractmethod
    def is_done(self):
        """
        Return the done signal
        """
        raise NotImplementedError
