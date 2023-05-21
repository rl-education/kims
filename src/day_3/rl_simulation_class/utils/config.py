import os
from dataclasses import dataclass
from typing import Any

import yaml
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class CartpolePhysics:
    override_usd_defaults: bool = False
    enable_self_collisions: bool = False
    enable_gyroscopic_forces: bool = True
    # per-actor
    solver_position_iteration_count: int = 4
    solver_velocity_iteration_count: int = 0
    sleep_threshold: float = 0.005
    stabilization_threshold: float = 0.001
    # per-body
    density: float = -1
    max_depenetration_velocity: float = 100.0
    # per-shape
    contact_offset: float = 0.02
    rest_offset: float = 0.001


@dataclass_json
@dataclass
class CartpoleTask:
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
    name: str = "cartpole"
    add_ground_plane: bool = True
    task: CartpoleTask = CartpoleTask()
    physics: CartpolePhysics = CartpolePhysics()


@dataclass_json
@dataclass
class PPOConfig:
    name: str = "ppo"
    hidden_size: int = 256
    learning_rate: float = 3e-4
    num_steps: int = 2000000
    batch_size: int = 4096 * 4
    mini_batch_size: int = 256
    ppo_epochs: int = 4
    gamma: float = 0.99
    tau: float = 0.95
    clip_param: float = 0.2


@dataclass_json
@dataclass
class Config:
    task_config_file: str
    policy_config_file: str
    task_config: Any = None
    policy_config: Any = None
    device: str = ""
    test: bool = False
    checkpoint: str = ""
    headless: bool = False


def load_config(config_path: str, config_file_name: str) -> Config:
    """Load a config from a yaml file."""
    with open(os.path.join(config_path, config_file_name), "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_dict(config_dict)
    if not config.device:
        import torch

        config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.task_config = load_task_config(config_path, config)
    config.policy_config = load_policy_config(config_path, config)
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
    if policy_config_dict["name"].lower() == "ppo":
        return PPOConfig.from_dict(policy_config_dict)
    raise ValueError(f"Unknown policy config: {policy_config_dict['name']}")
