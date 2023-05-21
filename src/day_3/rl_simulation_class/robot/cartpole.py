# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
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


from typing import Optional

import numpy as np
import omni.usd
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import PhysxSchema, UsdPhysics
from rl_simulation_class.utils.config import CartpolePhysics


class Cartpole(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Cartpole",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            self._usd_path = assets_root_path + "/Isaac/Robots/Cartpole/cartpole.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

    def apply_settings(self, config: CartpolePhysics):
        stage = omni.usd.get_context().get_stage()
        base_prim = get_prim_at_path(self.prim_path)
        prims = [base_prim]
        while len(prims) > 0:
            prim_tmp = prims.pop(0)

            children_prims = prim_tmp.GetPrim().GetChildren()
            prims = prims + children_prims

        # parse through all children prims
        prims = [base_prim]
        while len(prims) > 0:
            cur_prim = prims.pop(0)
            rigid_body = UsdPhysics.RigidBodyAPI.Get(stage, cur_prim.GetPath())
            collision_body = UsdPhysics.CollisionAPI.Get(stage, cur_prim.GetPath())
            articulation = UsdPhysics.ArticulationRootAPI.Get(stage, cur_prim.GetPath())
            if rigid_body:
                self.apply_rigid_body_settings(cur_prim, config)
            if collision_body:
                self.apply_rigid_shape_settings(cur_prim, config)
            if articulation:
                self.apply_articulation_settings(cur_prim, config)
            children_prims = cur_prim.GetPrim().GetChildren()
            prims = prims + children_prims

    def apply_articulation_settings(self, prim, config: CartpolePhysics):
        stage = omni.usd.get_context().get_stage()
        physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Get(stage, prim.GetPath())

        # enable self collisions
        enable_self_collisions = physx_articulation_api.GetEnabledSelfCollisionsAttr()
        if config.enable_self_collisions != -1:
            enable_self_collisions.Set(config.enable_self_collisions)

        self.set_articulation_position_iteration(prim, config.solver_position_iteration_count)
        self.set_articulation_velocity_iteration(prim, config.solver_velocity_iteration_count)
        self.set_articulation_sleep_threshold(prim, config.sleep_threshold)
        self.set_articulation_stabilization_threshold(prim, config.stabilization_threshold)

    def apply_rigid_shape_settings(self, prim, config: CartpolePhysics):
        self.set_contact_offset(prim, config.contact_offset)
        self.set_rest_offset(prim, config.rest_offset)

    def apply_rigid_body_settings(self, prim, config: CartpolePhysics):
        self.set_position_iteration(prim, config.solver_position_iteration_count)
        self.set_velocity_iteration(prim, config.solver_velocity_iteration_count)
        self.set_max_depenetration_velocity(prim, config.max_depenetration_velocity)
        self.set_sleep_threshold(prim, config.sleep_threshold)
        self.set_stabilization_threshold(prim, config.stabilization_threshold)
        self.set_gyroscopic_forces(prim, config.enable_gyroscopic_forces)

    def set_contact_offset(self, prim, value=None):
        physx_collision_api = self._get_physx_collision_api(prim)
        contact_offset = physx_collision_api.GetContactOffsetAttr()
        if value is not None and value != -1:
            contact_offset.Set(value)

    def set_rest_offset(self, prim, value=None):
        physx_collision_api = self._get_physx_collision_api(prim)
        rest_offset = physx_collision_api.GetRestOffsetAttr()
        if value is not None and value != -1:
            rest_offset.Set(value)

    def set_position_iteration(self, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        solver_position_iteration_count = physx_rb_api.GetSolverPositionIterationCountAttr()
        if value is not None and value != -1:
            solver_position_iteration_count.Set(value)

    def set_velocity_iteration(self, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        solver_velocity_iteration_count = physx_rb_api.GetSolverVelocityIterationCountAttr()
        if value is not None and value != -1:
            solver_velocity_iteration_count.Set(value)

    def set_max_depenetration_velocity(self, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        max_depenetration_velocity = physx_rb_api.GetMaxDepenetrationVelocityAttr()
        if value is not None and value != -1:
            max_depenetration_velocity.Set(value)

    def set_sleep_threshold(self, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        sleep_threshold = physx_rb_api.GetSleepThresholdAttr()
        if value is not None and value != -1:
            sleep_threshold.Set(value)

    def set_stabilization_threshold(self, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        stabilization_threshold = physx_rb_api.GetStabilizationThresholdAttr()
        if value is not None and value != -1:
            stabilization_threshold.Set(value)

    def set_gyroscopic_forces(self, prim, value=None):
        physx_rb_api = self._get_physx_rigid_body_api(prim)
        enable_gyroscopic_forces = physx_rb_api.GetEnableGyroscopicForcesAttr()
        if value is not None and value != -1:
            enable_gyroscopic_forces.Set(value)

    def set_articulation_position_iteration(self, prim, value):
        arti_api = self._get_physx_articulation_api(prim)
        solver_position_iteration_count = arti_api.GetSolverPositionIterationCountAttr()
        if value is not None and value != -1:
            solver_position_iteration_count.Set(value)

    def set_articulation_velocity_iteration(self, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        solver_velocity_iteration_count = arti_api.GetSolverVelocityIterationCountAttr()
        if value is not None and value != -1:
            solver_velocity_iteration_count.Set(value)

    def set_articulation_sleep_threshold(self, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        sleep_threshold = arti_api.GetSleepThresholdAttr()
        if value is not None and value != -1:
            sleep_threshold.Set(value)

    def set_articulation_stabilization_threshold(self, prim, value=None):
        arti_api = self._get_physx_articulation_api(prim)
        stabilization_threshold = arti_api.GetStabilizationThresholdAttr()
        if value is not None and value != -1:
            stabilization_threshold.Set(value)

    def _get_physx_articulation_api(self, prim):
        arti_api = PhysxSchema.PhysxArticulationAPI(prim)
        if not arti_api:
            arti_api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
        return arti_api

    def _get_physx_rigid_body_api(self, prim):
        physx_rb_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        if not physx_rb_api:
            physx_rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        return physx_rb_api

    def _get_physx_collision_api(self, prim):
        physx_collision_api = PhysxSchema.PhysxCollisionAPI(prim)
        if not physx_collision_api:
            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        return physx_collision_api
