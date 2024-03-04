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
            0.63731742,
            -0.06401012,
            0.0715,  # 0.073 for standard pose from move to PCB script
            3.09726181,
            -0.00431366,
            0.00620241
        ]
    )
    # currpose = requests.post(url + "getpos").json()['pose']
    # currpose = [0.6408740949404622,-0.06379811037824235,0.06944927438130806,0.9998948294098409,0.003954085211026688,-0.004827440406522266,0.013091644233279227]
    # target_posn = currpose[:3]
    # target_quat = currpose[3:]
    # target_euler = quat_2_euler(target_quat)
    # TARGET_POSE = np.concatenate([target_posn, target_euler])
    RESET_POSE = TARGET_POSE + np.array([0.0, 0.0, 0.04, 0.0, 0.0, 0.0])
    REWARD_THRESHOLD: np.ndarray = np.array([0.003, 0.003, 0.001, 0.1, 0.1, 0.1])
    ACTION_SCALE = np.array([0.02, 0.2, 1])
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
        "translational_clip_neg_z": 0.001,
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
