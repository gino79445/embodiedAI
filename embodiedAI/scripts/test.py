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


import numpy as np
import hydra
from omegaconf import DictConfig

from embodiedAI.utils.hydra_cfg.hydra_utils import *
from embodiedAI.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from embodiedAI.utils.task_util import initialize_task
from embodiedAI.envs.env import env as ENV
from controller import controller 

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless
    render = cfg.render
    sim_app_cfg_path = cfg.sim_app_cfg_path
    env = ENV(headless=headless,render=render,sim_app_cfg_path=sim_app_cfg_path)
    task = initialize_task(cfg_dict, env)
    env.reset()
    episode_reward = 0
    env.step(np.array([0,0,0,0,0,0,0,0,0,0,0,0]))
#    env.step(np.array([0,0,0,200,200,-223,2,0,0,0,0,0]))
#    env.step(np.array([0,100,0,0,0,0,0,0,0,0,0,0]))
#    env.step(np.array([-100,0,0,0,0,0,0,0,0,0,0,0]))
#    env.step(np.array([120,0,0,0,0,0,0,0,0,0,0,0]))
#    env.step(np.array([0,200,0,0,0,0,0,0,0,0,0,0]))
    #api = controller(env)
   # api.forward_backward(10)
   # api.rotation(3.14)
   # api.forward_backward(-10)
   # api.rotation(-3.14)
   # api.arm(np.array([1,0,0,-10,2,2,0]))
   # api.gripper(-5)
   # api.gripper(5)
    #api.stop()
if __name__ == '__main__':
    parse_hydra_configs()
