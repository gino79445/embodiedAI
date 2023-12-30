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

import torch
#from embodiedAI.tasks.base.rl_task import RLTask
from embodiedAI.handlers.tiagodualWBhandler import TiagoDualWBHandler
from omni.isaac.core.tasks import BaseTask
from embodiedAI.tasks.base.base import Base
# from omni.isaac.core.utils.prims import get_prim_at_path
# from omni.isaac.core.utils.prims import create_prim
# from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.isaac_sensor import _isaac_sensor
from embodiedAI.tasks.utils import scene_utils


# Whole Body example task with holonomic robot base
class TiagoDualWBNavman(Base):
    def __init__(
        self,
        name,
        sim_config,
        env
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._device = self._cfg["sim_device"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        
        self._gamma = self._task_cfg["env"]["gamma"]
        self._max_episode_length = self._task_cfg["env"]["horizon"]
        
        self._randomize_on_reset = self._task_cfg["env"]["randomize_robot_on_reset"]

        # Choose num_obs and num_actions based on task
        # (7 right arm dofs pos + vel * 2) + 3 holo base pos + vel * 2 = 10*2 = 20
        self._num_observations = 20
        self._move_group = self._task_cfg["env"]["move_group"]
        # Velocity control. Actions are arm (7) and base (3) velocities
        self._num_actions = 12
        # env specific velocity limits
        # self.max_arm_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        self.max_rot_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        self.max_base_xy_vel = torch.tensor(self._task_cfg["env"]["max_base_xy_vel"], device=self._device)

        # Environment object settings: (reset() randomizes the environment)
        self._obstacle_names = ["mammut1", "godishus","mammut2","mammut3","mammut4","mammut5","mammut6","mammut7","mammut8"] # ShapeNet models in usd format
        self._tabular_obstacle_mask = [True, True,True] # Mask to denote which objects are tabular (i.e. grasp objects can be placed on them)
        self._grasp_obj_names = ["004_sugar_box", "008_pudding_box", "010_potted_meat_can", "061_foam_brick"] # YCB models in usd format
        self._num_obstacles = min(self._task_cfg["env"]["num_obstacles"],len(self._obstacle_names))
        self._num_grasp_objs = min(self._task_cfg["env"]["num_grasp_objects"],len(self._grasp_obj_names))
        self._obj_states = torch.zeros((6*(self._num_obstacles+self._num_grasp_objs-1),self._num_envs),device=self._device) # All grasp objs except the target object will be used in obj state (BBox)
        self._obstacles = []
        self._obstacles_dimensions = []
        self._grasp_objs = []
        self._grasp_objs_dimensions = []
        #  Contact sensor interface for collision detection:
        self._contact_sensor_interface = _isaac_sensor.acquire_contact_sensor_interface()
        self.sence = ["Kitchen_set"]

        # Handler for Tiago
        self.tiago_handler = TiagoDualWBHandler(move_group=self._move_group,use_torso= False ,sim_config=self._sim_config, num_envs=self._num_envs, device=self._device)

        Base.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        import omni
        self.tiago_handler.get_robot()
         # Spawn obstacles (from ShapeNet usd models):
        obst = scene_utils.sence(name="Kitchen_set", prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
        self._obstacles.append(obst) # Add to list of obstacles (Geometry Prims)
        for i in range(self._num_obstacles):
            if i ==0:
                pose = torch.tensor([0.0,50.0,0.0],device=self._device)
            elif i ==1:
                pose = torch.tensor([50.0,0.0,0.0],device=self._device)
            obst = scene_utils.spawn_obstacle(name=self._obstacle_names[i], prim_path=self.tiago_handler.default_zero_env_path, device=self._device,pose=pose) 
            self._obstacles.append(obst) # Add to list of obstacles (Geometry Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor", sensor_period=float(self._sim_config.task_config["sim"]["dt"]),parent=obst.prim_path)
        # Enable object axis-aligned bounding box computations
        scene.enable_bounding_boxes_computations()
        # Add spawned objects to scene registry and store their bounding boxes:
        for obst in self._obstacles:
            scene.add(obst)
            self._obstacles_dimensions.append(scene.compute_object_AABB(obst.name)) # Axis aligned bounding box used as dimensions

        super().set_up_scene(scene)
        self._robots = self.tiago_handler.create_articulation_view()
        scene.add(self._robots)
        # Optional viewport for rendering in a separate viewer
        import omni.kit
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        self.viewport_window = omni.kit.viewport_legacy.get_default_viewport_window()
        self.sd_helper = SyntheticDataHelper()
        self.sd_helper.initialize(sensor_names=["rgb"], viewport=self.viewport_window)

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        self.tiago_handler.post_reset()

    def get_observations(self):
        # Get robot data: joint positions and velocities
        robot_pos_vels = self.tiago_handler.get_robot_obs()
        # TODO: Scale robot observations as per env
        self.obs_buf = robot_pos_vels
        return self.obs_buf

    def get_render(self):
        # Get ground truth viewport rgb image
        gt = self.sd_helper.get_groundtruth(
            ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        )
        return gt["rgb"][:, :, :3]
    
    def pre_physics_step(self, actions) -> None:
        # actions (num_envs, num_action)
        # Handle resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Scale and clip the actions (velocities) before sending to robot
        actions = torch.clamp(actions, -1, 1)
        actions *= self.max_rot_vel
        actions[:,-3:-1] *= (self.max_base_xy_vel/self.max_rot_vel) # scale base xy velocity joint velocities
        # NOTE: actions shape has to match the move_group selected
        #self.tiago_handler.get_lidar()
        self.tiago_handler.apply_actions(actions)
    
    # def pre_physics_step(self, actions) -> None:
    #     # make the robot move with the actions (distances)
    #     self.tiago_handler.move_robot_to_distance()
        


    def reset_idx(self, env_ids):
        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        # reset dof values
        self.tiago_handler.reset(indices,randomize=self._randomize_on_reset)
        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.extras[env_ids] = 0

    def calculate_metrics(self) -> None:
        # data from obs buffer is available (get_observations() called before this function)
        test = self.obs_buf[:,0]
        wp = self._robots.get_world_poses() # gets only root prim poses
        reward = torch.abs(test)
        # reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
        self.rew_buf[:] = reward
        self.extras = torch.where(reward >= 1.0, 1, self.extras)

    def is_done(self) -> None:
        # cart_pos = self.obs_buf[:, 0]
        # pole_pos = self.obs_buf[:, 2]
        # cond = True
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.zeros(self._num_envs, dtype=int, device=self._device)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets
