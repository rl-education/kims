import dataclasses
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from gym import Space, spaces
from omni.isaac.cloner import GridCloner
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import define_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.gym.vec_env import VecEnvBase
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner
from rl_simulation_class.utils.config import Config, RLGamesConfig


class MultiEnvRLTask(BaseTask, ABC):
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
        super().__init__(name=name, offset=offset)

        self._config = config
        self._device = self._config.device
        self.test = self._config.test
        print("Task Device:", self._device)

        self._env = env

        # initialize params in case they were not yet initialized by the derived class
        if not hasattr(self, "_num_envs"):
            self._num_envs = self._config.task_config.task.num_envs
        if not hasattr(self, "_num_actions"):
            self._num_actions = 1
        if not hasattr(self, "_num_observations"):
            self._num_observations = 1
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

        self._num_envs = self._config.task_config.task.num_envs
        self._env_spacing = self._config.task_config.task.env_spacing

        ####### Solution 1: Create a GridCloner object #######
        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self.default_base_path)
        ######################################################
        define_prim(f"{self.default_base_path}/env_0")

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

    def clone_environments(self) -> None:
        """Clones environments based on value provided in task config
        and applies collision filters to disable collisions across environments.
        """
        collision_filter_global_paths = [self._ground_plane_path]
        ####### Solution 1: Clone the environment using the GridCloner #######
        prim_paths = self._cloner.generate_paths(f"{self.default_base_path}/env", self._num_envs)
        self._env_pos = self._cloner.clone(
            source_prim_path=f"{self.default_base_path}/env_0",
            prim_paths=prim_paths,
        )
        #####################################################################
        self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)
        self._cloner.filter_collisions(
            self._env._world.get_physics_context().prim_path,
            "/World/collisions",
            prim_paths,
            collision_filter_global_paths,
        )

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


class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats."""

    def after_init(self, algo):
        """Called after the algorithm is initialized.

        Args:
            algo: the rl games algorithm
        """
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        """Process the infos from the env.

        Args:
            infos (dict): the info dict from the env
            done_indices: the indices of the envs that are done
        """
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if "episode" in infos:
                self.ep_infos.append(infos["episode"])

            if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if (
                        isinstance(v, float)
                        or isinstance(v, int)
                        or (isinstance(v, torch.Tensor) and len(v.shape) == 0)
                    ):
                        self.direct_info[k] = v

    def after_clear_stats(self):
        """Called after the stats are cleared."""
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        """Called after the stats are printed.

        Args:
            frame: the current frame
            epoch_num: the current epoch
            total_time: the total time elapsed
        """
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, epoch_num)
            self.ep_infos.clear()

        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"{k}/frame", v, frame)
            self.writer.add_scalar(f"{k}/iter", v, epoch_num)
            self.writer.add_scalar(f"{k}/time", v, total_time)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar("scores/mean", mean_scores, frame)
            self.writer.add_scalar("scores/iter", mean_scores, epoch_num)
            self.writer.add_scalar("scores/time", mean_scores, total_time)


class RLGPUEnv(vecenv.IVecEnv):
    """RLGPUEnv is a wrapper around the IVecEnv that allows us to use the VecEnvBase class."""

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = self.env.action_space
        info["observation_space"] = self.env.observation_space

        print(info["action_space"], info["observation_space"])

        return info


class RLGTrainer:
    """Class to run the RL Games trainer."""

    def __init__(self, env, config: RLGamesConfig):
        vecenv.register(
            "RLGPU",
            lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register(
            "rlgpu",
            {
                "vecenv_type": "RLGPU",
                "env_creator": lambda **kwargs: env,
            },
        )
        self.rlg_config_dict = dataclasses.asdict(config)

    def run(self, train: bool = True, checkpoint: str = None, sigma: float = None):
        # create runner and set the settings
        runner = Runner(RLGPUAlgoObserver())
        runner.load(self.rlg_config_dict)
        runner.reset()

        runner.run(
            {
                "train": train,
                "play": not train,
                "checkpoint": checkpoint,
                "sigma": sigma,
            },
        )
