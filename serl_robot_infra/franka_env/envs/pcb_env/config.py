import numpy as np
from franka_env.envs.franka_env import DefaultEnvConfig
from franka_env.utils.rotations import quat_2_euler

class PCBEnvConfig(DefaultEnvConfig):
    """Set the configuration for FrankaEnv."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": "218622274461",
        "wrist_2": "218622276379",
    }
    TARGET_POSE = np.array(
        [
            0.6378155575883534,
            -0.06302125722373966,
            0.073,
            3.12308208,
            -0.01842338,
            0.06769943,
        ]
    )
    # currpose = requests.post(url + "getpos").json()['pose']
    # sample currpose = [0.6378155575883534,-0.06302125722373966,0.07161134991397702,0.9993448317889017,0.033755161543103485,0.009519101764174228,0.008937726087444324]
    # sample currpose = [0.6370623577389048,-0.06343594171797502,0.07222356478376063,0.9998436663527301,0.0035192305543307423,-0.016541550334071255,0.005160909142885387]
    # sample currpose = [0.6362208204983791,-0.06280020351570716,0.07253667840286478,0.9998103195884877,0.011944632536439424,-0.008667952911610422,0.012708941350786162]
    # target_posn = currpose[:3]
    # target_quat = currpose[3:]
    # target_euler = quat_2_euler(target_quat)
    # TARGET_POSE = np.concatenate([target_posn, target_euler])
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
        "translational_Ki": 0,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "rotational_Ki": 0,
        "translational_clip_x": 0.0025,
        "translational_clip_y": 0.0015,
        "translational_clip_z": 0.002,
        "translational_clip_neg_x": 0.003,
        "translational_clip_neg_y": 0.002,
        "translational_clip_neg_z": 0.003,
        "rotational_clip_x": 0.025,
        "rotational_clip_y": 0.01,
        "rotational_clip_z": 0.01,
        "rotational_clip_neg_x": 0.025,
        "rotational_clip_neg_y": 0.01,
        "rotational_clip_neg_z": 0.01,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "translational_Ki": 0.1,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "rotational_Ki": 0.1,
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
    }
