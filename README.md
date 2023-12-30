# Learned Robot Placement

Based on the research paper: **Robot Learning of Mobile Manipulation With Reachability Behavior Priors** [1] [[Paper](https://arxiv.org/abs/2203.04051)] [[Project site](https://irosalab.com/rlmmbp/)] [[github](https://github.com/iROSA-lab/rlmmbp])]


## Installation
### Enviroment
python=3.7
isaac-sim (version 2022.1.0)

### Setup environments
```
cd learned_robot_placement
pip install -e .
```

## Experiments

### Launching the experiments
- Activate the conda environment:
    ```
    conda activate isaac-sim-lrp
    ```
- source the isaac-sim conda_setup file:
    ```
    source <PATH_TO_ISAAC_SIM>/isaac_sim-2022.1.0/setup_conda_env.sh
    ```
- test:
    ```
    cd learned_robot_placement
    python scripts/test.py
    ```

## Troubleshooting

- **"[Error] [omni.physx.plugin] PhysX error: PxRigidDynamic::setGlobalPose: pose is not valid."** This error can be **ignored** for now. Isaac-sim 2022.1.0 has some trouble handling the set_world_pose() function for RigidPrims, but this doesn't affect the experiments.
