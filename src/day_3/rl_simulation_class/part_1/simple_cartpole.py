import math
import os

import numpy as np
import torch
from gym import spaces
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.gym.vec_env import VecEnvBase
from rl_simulation_class.part_1.simple_rl_task import SimpleRLTask
from rl_simulation_class.utils.config import CartpoleConfig, Config


class CartpoleTask(SimpleRLTask):
    """Cartpole task for single environment RL training."""

    def __init__(self, name: str, config: Config, env: VecEnvBase, offset=None) -> None:
        self._config = config
        assert isinstance(self._config.task_config, CartpoleConfig)

        self._cartpole_position = torch.tensor([0.0, 0.0, 2.0])
        self._reset_dist = self._config.task_config.task.reset_dist
        self._max_push_effort = self._config.task_config.task.max_effort
        self._max_episode_length = self._config.task_config.task.max_episode_length

        self._num_observations = 4
        self._num_actions = 1
        self._action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self._observation_space = spaces.Box(
            np.ones(self._num_observations) * -np.Inf,
            np.ones(self._num_observations) * np.Inf,
        )

        self._cart_dof_idx = 0
        self._pole_dof_idx = 0

        super().__init__(name, env, self._config, offset)

    def set_up_scene(self, scene: Scene) -> None:
        """Method used to load the objects in the scene.

        Args:
            scene (Scene): The scene to load the objects into.
        """
        assert isinstance(self._config.task_config, CartpoleConfig)
        super().set_up_scene(scene)
        # retrieve file path for the Cartpole USD file
        if self._config.task_config.cartpole_file:
            usd_path = os.path.join(self._config.config_path, self._config.task_config.cartpole_file)
        else:
            assets_root_path = get_assets_root_path()
            usd_path = assets_root_path + "/Isaac/Robots/Cartpole/cartpole.usd"
        # add the Cartpole USD to our stage
        create_prim(
            prim_path=f"{self.default_base_path}/Cartpole",
            prim_type="Xform",
            position=self._cartpole_position,
        )
        add_reference_to_stage(usd_path, f"{self.default_base_path}/Cartpole")
        # create an ArticulationView wrapper for our cartpole - this can be extended towards accessing multiple cartpoles
        self._cartpoles = ArticulationView(
            prim_paths_expr=f"{self.default_base_path}/Cartpole*",
            name="cartpole_view",
        )
        # add Cartpole ArticulationView to the Scene
        scene.add(self._cartpoles)

    def cleanup(self) -> None:
        """Called before calling a reset() on the world. Reset data structures."""
        self._observations_buffer = torch.zeros(
            (1, self.num_observations),
            device=self._device,
            dtype=torch.float,
        )
        self._done_buffer = torch.ones(1, device=self._device, dtype=torch.long)
        self._episodes_count = torch.zeros(1, device=self._device, dtype=torch.long)

    def reset(self) -> None:
        """The reset function is called by the VecEnvBase class to reset the environment."""
        # set all environments to done
        self._done_buffer = torch.ones_like(self._done_buffer)
        reset_env_ids = self._done_buffer.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

    def pre_physics_step(self, actions: np.ndarray) -> None:
        """This function is called before the physics step is executed.

        Args:
            actions (np.ndarray): The actions to be executed.
        """
        if not self._env._world.is_playing():
            return

        # generate joint efforts from actions
        actions = torch.tensor(actions)

        forces = torch.zeros(
            (self._cartpoles.count, self._cartpoles.num_dof),
            dtype=torch.float32,
            device=self._device,
        )
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[0]

        # apply joint efforts
        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def get_observations(self) -> torch.Tensor:
        """Get the observations from the environment.

        Returns:
            torch.Tensor: A tensor containing the observations. It should be in the cpu device.
        """
        # get the positions for all the joints
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        # get the velocities for all the joints
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        # organize the cartpole positions and velocities
        cart_pos = dof_pos[:, self._cart_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]

        self._observations_buffer[:, 0] = cart_pos
        self._observations_buffer[:, 1] = cart_vel
        self._observations_buffer[:, 2] = pole_pos
        self._observations_buffer[:, 3] = pole_vel

        return self._observations_buffer.cpu()

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Reset the environment for the given environment ids.

        Args:
            env_ids (torch.Tensor): The environment ids to reset.
        """
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = (
            0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        )

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = (
            0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        )

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self._done_buffer[env_ids] = 0
        self._episodes_count[env_ids] = 0

    def post_reset(self):
        """This function is called after the environment is reset.
        We can use it to get the values which are only available after the environment is reset.
        """
        # get the dof indices for the cart and pole joints
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> float:
        """Generate the reward

        Returns:
            float: The reward for the current step.
        """
        cart_pos = self._observations_buffer[:, 0]
        cart_vel = self._observations_buffer[:, 1]
        pole_angle = self._observations_buffer[:, 2]
        pole_vel = self._observations_buffer[:, 3]

        # compute reward based on angle of pole and cart velocity
        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # apply a penalty if cart is too far from center
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # apply a penalty if pole is too far from upright
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        return reward.item()

    def is_done(self) -> bool:
        """returns true if the environment is done, false otherwise

        Returns:
            bool: True if the environment is done, false otherwise
        """
        cart_pos = self._observations_buffer[:, 0]
        pole_pos = self._observations_buffer[:, 2]

        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(self._episodes_count >= self._max_episode_length, 1, resets)
        self._done_buffer[:] = resets

        return self._done_buffer.item()
