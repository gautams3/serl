import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig


class PCBEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": "218622274461",
        "wrist_2": "218622276379",
    }
    TARGET_POSE = np.array(
        [
            0.5868045273591798,
            0.012909963287586323,
            0.06566930637555402,
            3.1405188,
            0.0,  # -0.041635,
            0.0,  # -0.0306968,
        ]
    )
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.04, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD = [0.003, 0.003, 0.001, 0.1, 0.1, 0.1]
    ACTION_SCALE = (0.02, 0.2, 1)
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.05
    RANDOM_RZ_RANGE = np.pi / 9
    ABS_POSE_LIMIT_LOW = np.array(
        [
            TARGET_POSE[0] - RANDOM_XY_RANGE,
            TARGET_POSE[1] - RANDOM_XY_RANGE,
            TARGET_POSE[2] - 0.005,
            TARGET_POSE[3] - 0.05,
            TARGET_POSE[4] - 0.05,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            TARGET_POSE[0] + RANDOM_XY_RANGE,
            TARGET_POSE[1] + RANDOM_XY_RANGE,
            TARGET_POSE[2] + 0.05,
            TARGET_POSE[3] + 0.05,
            TARGET_POSE[4] + 0.05,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 180,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0025,
        "translational_clip_y": 0.0015,
        "translational_clip_z": 0.002,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.002,
        "translational_clip_neg_z": 0.003,
        "rotational_clip_x": 0.025,
        "rotational_clip_y": 0.007,
        "rotational_clip_z": 0.01,
        "rotational_clip_neg_x": 0.025,
        "rotational_clip_neg_y": 0.007,
        "rotational_clip_neg_z": 0.01,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }
