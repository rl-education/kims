import math
from typing import Any, Tuple

import numpy as np
import torch
from gym import spaces
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.gym.vec_env import VecEnvBase
from rl_simulation_class.robot.cartpole import Cartpole
from rl_simulation_class.tasks.multi_env_rl_task import MultiEnvRLTask
from rl_simulation_class.utils.config import CartpoleConfig, Config


class CartpoleTask(MultiEnvRLTask):
    def __init__(self, name: str, config: Config, env: VecEnvBase, offset=None) -> None:
        self._config = config
        assert isinstance(self._config.task_config, CartpoleConfig)

        self._cartpole_position = torch.tensor([0.0, 0.0, 2.0])
        self._reset_dist = self._config.task_config.task.reset_dist
        self._max_push_effort = self._config.task_config.task.max_effort
        self._max_episode_length = self._config.task_config.task.max_episode_length
        self.clip_obs = self._config.task_config.task.clip_observations
        self.clip_actions = self._config.task_config.task.clip_actions
        self.control_frequency_inv = self._config.task_config.task.control_frequency_inv

        self._num_observations = 4
        self._num_actions = 1
        self._action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self._observation_space = spaces.Box(
            np.ones(self._num_observations) * -np.Inf,
            np.ones(self._num_observations) * np.Inf,
        )

        self._cart_dof_idx = 0
        self._pole_dof_idx = 0

        self._do_reset = True

        super().__init__(name, env, self._config, offset)

    def set_up_scene(self, scene) -> None:
        assert isinstance(self._config.task_config, CartpoleConfig)
        cartpole = Cartpole(
            prim_path=f"{self.default_base_env_path}/env_0/Cartpole",
            name="Cartpole",
            translation=self._cartpole_position,
        )
        # cartpole.apply_settings(self._config.task_config.physics)
        super().set_up_scene(scene)
        self._cartpoles = ArticulationView(
            prim_paths_expr=f"{self.default_base_env_path}/.*/Cartpole",
            name="cartpole_view",
            reset_xform_properties=False,
        )
        scene.add(self._cartpoles)

    def get_observations(self) -> dict:
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        cart_pos = dof_pos[:, self._cart_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]

        self._observations_buffer[:, 0] = cart_pos
        self._observations_buffer[:, 1] = cart_vel
        self._observations_buffer[:, 2] = pole_pos
        self._observations_buffer[:, 3] = pole_vel

        self._observations_buffer = (
            torch.clamp(self._observations_buffer, -self.clip_obs, self.clip_obs).to(self._device).clone()
        )
        return self._observations_buffer

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self._done_buffer.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions).to(self._device).clone()

        forces = torch.zeros(
            (self._cartpoles.count, self._cartpoles.num_dof),
            dtype=torch.float32,
            device=self._device,
        )
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids):
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
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        if self._do_reset:
            indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
            self.reset_idx(indices)

    def calculate_metrics(self) -> None:
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

        self._rewards_buffer[:] = reward

        return self._rewards_buffer

    def is_done(self) -> None:
        cart_pos = self._observations_buffer[:, 0]
        pole_pos = self._observations_buffer[:, 2]

        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(self._episodes_count >= self._max_episode_length, 1, resets)
        self._done_buffer[:] = resets

        return self._done_buffer

    def run_test(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        self._do_reset = False
        ret = self._env.step(action)
        self._do_reset = True
        return ret
