from abc import ABC
from typing import Optional

import numpy as np
import torch
from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.gym.vec_env import VecEnvBase
from rl_simulation_class.tasks.simple_rl_task import SimpleRLTask
from rl_simulation_class.utils.config import Config


class MultiEnvRLTask(SimpleRLTask, ABC):

    """
    Multi Environment Interface for RL tasks.
    """

    def __init__(
        self,
        name: str,
        env: VecEnvBase,
        config: Config,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        """Multi Environment Interface for RL tasks.

        Args:
            name (str): name of the task. Needs to be unique.
            env (VecEnvBase): the RL environment
            config (Config): the task configuration
            offset (np.Optional[ndarray], optional): offset applied to all assets of the task. Defaults to None.
        """

        super().__init__(name=name, env=env, config=config, offset=offset)

        self._num_envs = self._config.task_config.task.num_envs
        self._env_spacing = self._config.task_config.task.env_spacing

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_env_path)
        define_prim(f"{self.default_base_env_path}/env_0")

        self.cleanup()

    def cleanup(self) -> None:
        """Called before calling a reset() on the world. Reset data structures."""
        self._observations_buffer = torch.zeros(
            (self._num_envs, self.num_observations),
            device=self._device,
            dtype=torch.float,
        )
        self._rewards_buffer = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        self._done_buffer = torch.ones(self._num_envs, device=self._device, dtype=torch.long)
        self._episodes_count = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

    def set_up_scene(self, scene, replicate_physics=True) -> None:
        """Clones environments based on value provided in task config and applies collision filters to mask
            collisions across environments.

        Args:
            scene (Scene): Scene to add objects to.
            replicate_physics (bool): Clone physics using PhysX API for better performance
        """

        super().set_up_scene(scene)

        collision_filter_global_paths = list()
        if self._ground_plane_path is not None:
            collision_filter_global_paths.append(self._ground_plane_path)
        prim_paths = self._cloner.generate_paths(f"{self.default_base_env_path}/env", self._num_envs)
        self._env_pos = self._cloner.clone(
            source_prim_path=f"{self.default_base_env_path}/env_0",
            prim_paths=prim_paths,
            replicate_physics=replicate_physics,
        )
        self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
        self._cloner.filter_collisions(
            self._env._world.get_physics_context().prim_path,
            "/World/collisions",
            prim_paths,
            collision_filter_global_paths,
        )
