import numpy as np
import gymnasium as gym
import time
import requests
import copy

from franka_env.envs.franka_env import FrankaEnv
from franka_env.utils.rotations import euler_2_quat
from franka_env.envs.pcb_env.config import PCBEnvConfig


class FrankaPCBInsert(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, config=PCBEnvConfig)

    def crop_image(self, name, image):
        """Crop realsense images to be a square."""
        if name == "wrist_1":
            return image[90:390, 170:470, :]
        elif name == "wrist_2":
            return image[90:390, 170:470, :]
        else:
            return ValueError(f"Camera {name} not recognized in cropping")

    def go_to_rest(self, jpos=False):
        requests.post(self.url + "pcb_compliance_mode")
        self.update_currpos()
        self._send_pos_command(self.clip_safety_box(self.currpos))
        time.sleep(0.5)

        requests.post(self.url + "pcb_compliance_mode")
        self.update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] += 0.03
        self.interpolate_move(reset_pose, timeout=1.5)

        requests.post(self.url + "precision_mode")
        time.sleep(1)  # wait for mode switching
        reset_pose = self.resetpos.copy()
        self.interpolate_move(reset_pose, timeout=1)

        # perform random reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._TARGET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self.interpolate_move(reset_pose, timeout=1.5)

        if jpos:
            requests.post(self.url + "precision_mode")
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)
            self.interpolate_move(self.resetpos, timeout=5)

        requests.post(self.url + "pcb_compliance_mode")
        return True