# Omniverse Isaac Gym Reinforcement Learning Environments for Isaac Sim

# BSD 3-Clause License

# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# Note: Licenses for assets such as Robots and Props used within these
# environments can be found inside their respective folders on the Nucleus
# server where they are hosted

import os
from enum import Enum

import carb
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.kit import SimulationApp


class IsaacLaunchType(Enum):
    """Enum class for specifying the type of Isaac application to launch."""

    HEADLESS = "headless"
    RENDER = "render"
    LIVESTREAM = "livestream"


class VecEnvBaseLivestream(VecEnvBase):
    """This classes overrides the VecEnvBase class to enable livestreaming of the simulation for Isaac 2022.1.1."""

    def __init__(
        self,
        launch_type: IsaacLaunchType = IsaacLaunchType.HEADLESS,
        sim_device: int = 0,
    ) -> None:
        """Initializes RL and task parameters.

        Args:
            launch_type (IsaacLaunchType): Type of Isaac application to launch. Defaults to IsaacLaunchType.HEADLESS.
            sim_device (int): GPU device ID for running physics simulation. Defaults to 0.
        """
        experience = ""
        if launch_type == IsaacLaunchType.HEADLESS:
            experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'

        self._simulation_app = SimulationApp(
            {"headless": launch_type is not IsaacLaunchType.RENDER, "physics_device": sim_device},
            experience=experience,
        )
        carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)
        self._render = launch_type is not IsaacLaunchType.HEADLESS
        self.sim_frame_count = 0
        if launch_type == IsaacLaunchType.LIVESTREAM:
            self.enable_livestream()

    def enable_livestream(self) -> None:
        """Enables livestreaming of the simulation."""
        from omni.isaac.core.utils.extensions import enable_extension

        self._simulation_app.set_setting("/app/livestream/enabled", True)
        self._simulation_app.set_setting("/app/window/drawMouse", True)
        self._simulation_app.set_setting("/app/livestream/proto", "ws")
        self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
        self._simulation_app.set_setting("/ngx/enabled", False)
        enable_extension("omni.kit.livestream.native")
        enable_extension("omni.services.streaming.manager")
