import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml
from dataclasses_json import dataclass_json
from rl_simulation_class.utils.vec_env import IsaacLaunchType


@dataclass_json
@dataclass
class CartpoleTask:
    """Cartpole task configuration."""

    num_envs: int = 1
    env_spacing: float = 4.0
    reset_dist: float = 3.0
    max_effort: float = 400.0
    max_episode_length: int = 500

    clip_observations: float = 5.0
    clip_actions: float = 1.0
    control_frequency_inv: float = 2  # 60 Hz


@dataclass_json
@dataclass
class CartpoleConfig:
    """Cartpole Environment configuration."""

    name: str = "cartpole"
    cartpole_file: str = ""
    add_ground_plane: bool = True
    task: CartpoleTask = CartpoleTask()


@dataclass_json
@dataclass
class RLGamesConfig:
    """RL Games Policy configuration."""

    name: str = "rl-games"
    params: Dict[Any, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class SB3Config:
    """Stable Baselines 3 Policy configuration."""

    name: str = "sb3"
    policy: str = "MlpPolicy"
    max_steps: int = 10000
    n_steps: int = 1000
    batch_size: int = 1000
    n_epochs: int = 20
    learning_rate: float = 0.001
    gamma: float = 0.99
    device: str = "cuda"
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    verbose: int = 1
    tensorboard_log: str = "./cartpole_tensorboard"


@dataclass_json
@dataclass
class Config:
    """Configuration for the RL simulation class."""

    task_config_file: str
    policy_config_file: str
    config_path: str = ""
    task_config: Any = None
    policy_config: Any = None
    device: str = ""
    test: bool = False
    checkpoint: str = ""
    launch_type: IsaacLaunchType = IsaacLaunchType.HEADLESS
    num_envs: int = 1


def load_config(config_path: str, config_file_name: str) -> Config:
    """Load a config from a yaml file."""
    with open(os.path.join(config_path, config_file_name), "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config: Config = Config.from_dict(config_dict)
    config.config_path = config_path
    if not config.device:
        import torch

        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.task_config = load_task_config(config_path, config)
    if isinstance(config.task_config, CartpoleConfig):
        config.task_config.task.num_envs = config.num_envs
    config.policy_config = load_policy_config(config_path, config)
    if isinstance(config.policy_config, RLGamesConfig) and "config" in config.policy_config.params:
        config.policy_config.params["config"]["device"] = config.device
        config.policy_config.params["config"]["device_name"] = config.device
        config.policy_config.params["config"]["num_actors"] = config.num_envs
    elif isinstance(config.policy_config, SB3Config):
        config.policy_config.device = config.device
    return config


def load_task_config(config_path: str, config: Config) -> Any:
    """Load a task config from a yaml file."""
    with open(os.path.join(config_path, config.task_config_file), "r", encoding="utf-8") as f:
        task_config_dict = yaml.safe_load(f)
    if task_config_dict["name"].lower() == "cartpole":
        return CartpoleConfig.from_dict(task_config_dict)
    raise ValueError(f"Unknown task config: {task_config_dict['name']}")


def load_policy_config(config_path: str, config: Config) -> Any:
    """Load a policy config from a yaml file."""
    with open(os.path.join(config_path, config.policy_config_file), "r", encoding="utf-8") as f:
        policy_config_dict = yaml.safe_load(f)
    if policy_config_dict["name"].lower() == "sb3":
        return SB3Config.from_dict(policy_config_dict)
    if policy_config_dict["name"].lower() == "rl-games":
        return RLGamesConfig.from_dict(policy_config_dict)
    raise ValueError(f"Unknown policy config: {policy_config_dict['name']}")
