from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from gym import Space, spaces
from omni.isaac.core.tasks import BaseTask
from omni.isaac.gym.vec_env import VecEnvBase
from omni.kit.viewport.utility import get_viewport_from_window_name

# import omni.kit
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf
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
            (1, self.num_observations),
            device=self._device,
            dtype=torch.float,
        )
        self._rewards_buffer = torch.zeros(1, device=self._device, dtype=torch.float)
        self._done_buffer = torch.ones(1, device=self._device, dtype=torch.long)
        self._episodes_count = torch.zeros(1, device=self._device, dtype=torch.long)

    def set_up_scene(self, scene) -> None:
        """Clones environments based on value provided in task config and applies collision filters to mask
            collisions across environments.

        Args:
            scene (Scene): Scene to add objects to.
            replicate_physics (bool): Clone physics using PhysX API for better performance
        """
        super().set_up_scene(scene)

        collision_filter_global_paths = list()
        if self._config.task_config.add_ground_plane:
            self._ground_plane_path = "/World/defaultGroundPlane"
            collision_filter_global_paths.append(self._ground_plane_path)
            scene.add_default_ground_plane(prim_path=self._ground_plane_path)

        self.set_initial_camera_params(camera_position=[10, 10, 3], camera_target=[0, 0, 0])

    def set_initial_camera_params(self, camera_position=[10, 10, 3], camera_target=[0, 0, 0]):
        if self._env._render:
            viewport_api_2 = get_viewport_from_window_name("Viewport")
            viewport_api_2.set_active_camera("/OmniverseKit_Persp")
            camera_state = ViewportCameraState("/OmniverseKit_Persp", viewport_api_2)
            camera_state.set_position_world(
                Gf.Vec3d(camera_position[0], camera_position[1], camera_position[2]),
                True,
            )
            camera_state.set_target_world(
                Gf.Vec3d(camera_target[0], camera_target[1], camera_target[2]),
                True,
            )

    @property
    def default_base_env_path(self):
        """Retrieves default path to the parent of all env prims.

        Returns:
            default_base_env_path(str): Defaults to "/World/envs".
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

    def reset(self):
        """Flags all environments for reset."""
        self._done_buffer = torch.ones_like(self._done_buffer)

    ### Methods called by the VecEnvBase
    @abstractmethod
    def pre_physics_step(self, actions):
        """Optionally implemented by individual task classes to process actions.

        Args:
            actions (torch.Tensor): Actions generated by RL policy.
        """
        raise NotImplementedError

    def get_observations(self) -> torch.Tensor:
        """
        Return the observations

        Returns:
            (torch.Tensor): Tensor of observation data.
        """
        return self._observations_buffer

    def calculate_metrics(self) -> torch.Tensor:
        """
        Return the rewards

        Returns:
            (torch.Tensor): Tensor of reward data.
        """
        return self._rewards_buffer

    def is_done(self) -> torch.Tensor:
        """
        Return the done signal

        Returns:
            (torch.Tensor): Tensor of done data.
        """
        return self._done_buffer
