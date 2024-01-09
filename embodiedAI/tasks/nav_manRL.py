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
import numpy as np
from embodiedAI.tasks.base.rl_task import RLTask
from embodiedAI.handlers.tiagodualWBhandler import TiagoDualWBHandler
from omni.isaac.core.objects.cone import VisualCone
from omni.isaac.core.prims import GeometryPrimView
from embodiedAI.tasks.utils.pinoc_utils import PinTiagoIKSolver # For IK

# from omni.isaac.core.utils.prims import get_prim_at_path
# from omni.isaac.core.utils.prims import create_prim
# from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.isaac_sensor import _isaac_sensor
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats, quat_diff_rad
from scipy.spatial.transform import Rotation
from embodiedAI.tasks.utils.usd_utils import create_area_light
from embodiedAI.tasks.utils import scene_utils
# Simple base placement environment for reaching
class TiagoDualWBNavmanRL(RLTask):
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
        
        self._randomize_robot_on_reset = self._task_cfg["env"]["randomize_robot_on_reset"]
        # Choose num_obs and num_actions based on task
        # 6D goal pose only (3 pos + 4 quat = 7)
        self._num_observations = 7
        self._move_group = self._task_cfg["env"]["move_group"]
        self._use_torso = self._task_cfg["env"]["use_torso"]
        # Position control. Actions are base SE2 pose (3) and discrete arm activation (2)
        self._num_actions = self._task_cfg["env"]["continous_actions"] + self._task_cfg["env"]["discrete_actions"]
        # env specific limits
        self._world_xy_radius = self._task_cfg["env"]["world_xy_radius"]
        self._action_xy_radius = self._task_cfg["env"]["action_xy_radius"]
        self._action_ang_lim = self._task_cfg["env"]["action_ang_lim"]
        # self.max_arm_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        # self.max_rot_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        # self.max_base_xy_vel = torch.tensor(self._task_cfg["env"]["max_base_xy_vel"], device=self._device)
        
        # End-effector reaching goal settings (reset() randomizes the goal)
        # Goal is 6D pose (metres, rotation in quaternion: 7 dimensions)
        self._goal_z_lim = self._task_cfg["env"]["goal_z_lim"]
        self._goal_lims = torch.tensor([[-self._world_xy_radius,-self._world_xy_radius,self._goal_z_lim[0],-np.pi,-np.pi,-np.pi],
                                        [ self._world_xy_radius, self._world_xy_radius,self._goal_z_lim[1], np.pi, np.pi, np.pi]], device=self._device)
        self._goal_distribution = torch.distributions.Uniform(self._goal_lims[0], self._goal_lims[1])
        goals_sample = self._goal_distribution.sample((self.num_envs,))
        self._goals = torch.hstack((torch.tensor([[0.8,0.0,0.4+0.15]]),euler_angles_to_quats(torch.tensor([[0.19635, 1.375, 0.19635]]),device=self._device)))[0].repeat(self.num_envs,1)
        # self._goals = torch.hstack((goals_sample[:,:3],euler_angles_to_quats(goals_sample[:,3:6],device=self._device)))
        self._goal_tf = torch.zeros((4,4),device=self._device)
        self._goal_tf[:3,:3] = torch.tensor(Rotation.from_quat(np.array([self._goals[0,3+1],self._goals[0,3+2],self._goals[0,3+3],self._goals[0,3]])).as_matrix(),dtype=float,device=self._device) # Quaternion in scalar last format!!!
        self._goal_tf[:,-1] = torch.tensor([self._goals[0,0], self._goals[0,1], self._goals[0,2], 1.0],device=self._device) # x,y,z,1
        self._curr_goal_tf = self._goal_tf.clone()
        self._goals_xy_dist = torch.linalg.norm(self._goals[:,0:2],dim=1)  # distance from origin
        self._goal_pos_threshold = self._task_cfg["env"]["goal_pos_thresh"]
        self._goal_ang_threshold = self._task_cfg["env"]["goal_ang_thresh"]



        # Reward settings
        self._reward_success = self._task_cfg["env"]["reward_success"]
        self._reward_dist_weight = self._task_cfg["env"]["reward_dist_weight"]
        self._reward_noIK = self._task_cfg["env"]["reward_noIK"]
        # self._reward_timeout = self._task_cfg["env"]["reward_timeout"]
        # self._reward_collision = self._task_cfg["env"]["reward_collision"]
        self._reward_collision = self._task_cfg["env"]["reward_collision"]
        self._collided = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._ik_fails = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._is_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # Get dt for integrating velocity commands and checking limit violations
        self._dt = torch.tensor(self._sim_config.task_config["sim"]["dt"]*self._sim_config.task_config["env"]["controlFrequencyInv"],device=self._device)

        self.max_rot_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        self.max_base_xy_vel = torch.tensor(self._task_cfg["env"]["max_base_xy_vel"], device=self._device)
        

        # IK solver
        self._ik_solver = PinTiagoIKSolver(move_group=self._move_group, include_torso=self._use_torso, include_base=False, max_rot_vel=100.0) # No max rot vel

        
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
        self.sence = ["Kitchen_set"]

        #  Contact sensor interface for collision detection:
        self._contact_sensor_interface = _isaac_sensor.acquire_contact_sensor_interface()


        # Handler for Tiago
        self.tiago_handler = TiagoDualWBHandler(move_group=self._move_group, use_torso=self._use_torso, sim_config=self._sim_config, num_envs=self._num_envs, device=self._device)






        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        import omni
        self.tiago_handler.get_robot()

        # ADD kitchen set
#        obst = scene_utils.sence(name="Kitchen_set", prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
#        create_area_light()    
#
#
#        self._obstacles.append(obst) # Add to list of obstacles (Geometry Prims)
        for i in range(self._num_obstacles):
            obst = scene_utils.spawn_obstacle(name=self._obstacle_names[i], prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
            self._obstacles.append(obst) # Add to list of obstacles (Geometry Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor", sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                parent=obst.prim_path)
        # Spawn grasp objs (from YCB usd models):
        for i in range(self._num_grasp_objs):
            grasp_obj = scene_utils.spawn_grasp_object(name=self._grasp_obj_names[i], prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
            self._grasp_objs.append(grasp_obj) # Add to list of grasp objects (Rigid Prims)
            # Optional: Add contact sensors for collision detection. Covers whole body by default
            omni.kit.commands.execute("IsaacSensorCreateContactSensor", path="/Contact_Sensor", sensor_period=float(self._sim_config.task_config["sim"]["dt"]),
                parent=grasp_obj.prim_path)
        # Goal visualizer
        goal_viz = VisualCone(prim_path=self.tiago_handler.default_zero_env_path+"/goal",
                                radius=0.05,height=0.05,color=np.array([1.0,0.0,0.0]))
        super().set_up_scene(scene)
        self._robots = self.tiago_handler.create_articulation_view()
        scene.add(self._robots)
        self._goal_vizs = GeometryPrimView(prim_paths_expr="/World/envs/.*/goal",name="goal_viz")
        scene.add(self._goal_vizs)
        # Enable object axis-aligned bounding box computations
        scene.enable_bounding_boxes_computations()
        # Add spawned objects to scene registry and store their bounding boxes:
        for obst in self._obstacles:
            scene.add(obst)
            self._obstacles_dimensions.append(scene.compute_object_AABB(obst.name)) # Axis aligned bounding box used as dimensions
        for grasp_obj in self._grasp_objs:
            scene.add(grasp_obj)
            self._grasp_objs_dimensions.append(scene.compute_object_AABB(grasp_obj.name)) # Axis aligned bounding box used as dimensions

        # Optional viewport for rendering in a separate viewer
        import omni.kit
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        self.sd_helper = SyntheticDataHelper()
    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        self.tiago_handler.post_reset()

    def transform_to_world_coordinates(self, quatpose, local_point):
        from pyquaternion import Quaternion
        # Extract the world position and orientation from quatpose
        world_pos = quatpose[:3]
        world_orient = Quaternion(quatpose[3], quatpose[4], quatpose[5], quatpose[6])
        # Extract local position and orientation from local_point
        local_pos = local_point[:3]
        local_orient = Quaternion(local_point[3], local_point[4], local_point[5], local_point[6])

        # Rotate the local position to the world frame and add the world position
        world_coordinates = world_orient.rotate(local_pos) + world_pos

        # Combine the local orientation with the world orientation
        combined_orientation = world_orient * local_orient
        # transform to tensor
        world_coordinates = torch.tensor(world_coordinates,device=self._device)
        combined_orientation = np.array([ combined_orientation.imaginary[0],
                         combined_orientation.imaginary[1], combined_orientation.imaginary[2] ,combined_orientation.real])
        combined_orientation = torch.tensor(combined_orientation,device=self._device)
        return world_coordinates, combined_orientation

    def get_observations(self):
        # Handle any pending resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            

        
        #print(f"rgb_data: {self.rgb_data.shape}")

        # # Get robot observations
        robot_joint_pos = self.tiago_handler.get_robot_obs()
        # Fill observation buffer
        # Goal: 3D pos + rot_quaternion (3+4=7)
        robot_pos = np.array([self.tiago_handler._robot_pose[0],self.tiago_handler._robot_pose[1]])
        curr_goal_pos = self._curr_goal_tf[0:2,3]
        # calculate the distance between robot_pos and curr_goal_pos
        dist = torch.linalg.norm(torch.tensor(robot_pos,device=self._device).unsqueeze(dim=0) - curr_goal_pos,dim=1)


        curr_goal_pos = self._curr_goal_tf[0:3,3].unsqueeze(dim=0)
        curr_goal_quat = torch.tensor(Rotation.from_matrix(self._curr_goal_tf[:3,:3]).as_quat()[[3, 0, 1, 2]],dtype=torch.float,device=self._device).unsqueeze(dim=0)
        grasp_pos = self.tiago_handler.get_grasp_pos()
        # transform grasp tuple to tensor
        grasp_pos = torch.tensor(grasp_pos,device=self._device).unsqueeze(dim=0)
        self.obs_buf = torch.hstack((curr_goal_pos,curr_goal_quat))
        # caculate the distance between grasp_pos and grasp_pos
        dist = torch.linalg.norm(grasp_pos - self.obs_buf[:,:3],dim=1)
        #self.obs_buf = torch.tensor(self.rgb_data,device=self._device)
        #self.rgb_data = self.sd_helper.get_groundtruth(["rgb"], self.ego_viewport.get_viewport_window())["rgb"]
        #self.depth_data = self.sd_helper.get_groundtruth(["depth"], self.ego_viewport.get_viewport_window())["depth"]
        #print(self.rgb_data.shape)
        #print(self.depth_data.shape)
        #self.obs_buf = self.obs_buf.view(-1)
        return self.obs_buf

    def get_render(self):
        # Get ground truth viewport rgb image
    #    gt = self.sd_helper.get_groundtruth(
    #        ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
    #    )
    
        gt = self.sd_helper.get_groundtruth(["rgb"], self.ego_viewport.get_viewport_window())
        return np.array(gt["rgb"][:, :, :3])
    
    def pre_physics_step(self, actions) -> None:
        # actions (num_envs, num_action)
        # Handle resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
         
        if actions[:,6] != 0:
            quatpose = self.tiago_handler._robot_pose
            # translate actions to np array
            actions = actions.cpu().numpy()

             # manipulate actions
            curr_goal_pos = self._curr_goal_tf[0:3,3]
            curr_goal_quat = Rotation.from_matrix(self._curr_goal_tf[:3,:3]).as_quat()[[3, 0, 1, 2]]
            des_quat = np.array([actions[0,9],actions[0,6],actions[0,7],actions[0,8]])
            des_quat = torch.tensor(des_quat,device=self._device)
            success, ik_positions = self._ik_solver.solve_ik_pos_tiago(des_pos=actions[0,3:6], des_quat=des_quat,
                                            pos_threshold=self._goal_pos_threshold, angle_threshold=self._goal_ang_threshold, verbose=False)

            if success:
                #self._is_success[0] = 1 # Can be used for reward, termination
                # set upper body positions
                self.tiago_handler.set_upper_body_positions(jnt_positions=torch.tensor(np.array([ik_positions]),dtype=torch.float,device=self._device))
            else:
                pass
                #self._ik_fails[0] = 1 # Can be used for reward
        else:
            # Scale and clip the actions (velocities) before sending to robot
            actions = torch.clamp(actions, -1, 1)
            actions *= self.max_rot_vel
            actions[:,0:2] *= (self.max_base_xy_vel/self.max_rot_vel) # scale base xy velocity joint velocities
            # NOTE: actions shape has to match the move_group selected
            #self.tiago_handler.get_lidar()
            self.tiago_handler.apply_actions(actions)
    
    def reset_idx(self, env_ids):
        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        # reset dof values
        self.tiago_handler.reset(indices,randomize=self._randomize_robot_on_reset)
        # reset the scene objects (randomize), get target end-effector goal/grasp as well as oriented bounding boxes of all other objects
        self._curr_grasp_obj, self._goals[env_ids], self._obj_bboxes = scene_utils.setup_tabular_scene(
                                self, self._obstacles, self._tabular_obstacle_mask[0:self._num_obstacles], self._grasp_objs,
                                self._obstacles_dimensions, self._grasp_objs_dimensions, self._world_xy_radius, self._device)
        self._curr_obj_bboxes = self._obj_bboxes.clone()
        # self._goals[env_ids] = torch.hstack((goals_sample[:,:3],euler_angles_to_quats(goals_sample[:,3:6],device=self._device)))
        
        self._goal_tf = torch.zeros((4,4),device=self._device)
        goal_rot = Rotation.from_quat(np.array([self._goals[0,3+1],self._goals[0,3+2],self._goals[0,3+3],self._goals[0,3]])) # Quaternion in scalar last format!!!
        self._goal_tf[:3,:3] = torch.tensor(goal_rot.as_matrix(),dtype=float,device=self._device)
        self._goal_tf[:,-1] = torch.tensor([self._goals[0,0], self._goals[0,1], self._goals[0,2], 1.0],device=self._device) # x,y,z,1
        self._curr_goal_tf = self._goal_tf.clone()
        self._goals_xy_dist = torch.linalg.norm(self._goals[:,0:2],dim=1) # distance from origin
        # Pitch visualizer by 90 degrees for aesthetics
        goal_viz_rot = goal_rot * Rotation.from_euler("xyz", [0,np.pi/2.0,0])
        self._goal_vizs.set_world_poses(indices=indices,positions=self._goals[env_ids,:3],
                orientations=torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]],device=self._device).unsqueeze(dim=0))

        # bookkeeping
        self._is_success[env_ids] = 0
        self._ik_fails[env_ids] = 0
        self._collided[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.extras[env_ids] = 0


    def calculate_metrics(self) -> None:
        # assuming data from obs buffer is available (get_observations() called before this function)
        # Distance reward
        prev_goal_xy_dist = self._goals_xy_dist
        curr_goal_xy_dist = torch.linalg.norm(self.obs_buf[:,:2],dim=1)
        curr_goal_xy_dist = 0
        goal_xy_dist_reduction = torch.tensor(prev_goal_xy_dist - curr_goal_xy_dist)
        reward = self._reward_dist_weight*goal_xy_dist_reduction
        # print(f"Goal Dist reward: {reward}")
        self._goals_xy_dist = curr_goal_xy_dist

        # IK fail reward (penalty)
        reward += self._reward_noIK*self._ik_fails

        # Success reward
        reward += self._reward_success*self._is_success
        # print(f"Total reward: {reward}")
        self.rew_buf[:] = reward
        self.extras[:] = self._is_success.clone() # Track success

    def is_done(self) -> None:
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > np.pi / 2, 1, resets)
        # resets = torch.zeros(self._num_envs, dtype=int, device=self._device)
        
        # reset if success OR if reached max episode length
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, self._is_success)
        self.reset_buf[:] = resets
