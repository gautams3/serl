""" Scipt to move the transformer above the PCB. """

import requests
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=3, suppress=True)
SERVER_URL = "http://127.0.0.1:5000/"
# Fixed pick pose for hopper
ABOVE_PICK_POSE = [0.36043007, -0.05832796, 0.101, 1.0, 0.0, 0.0, 0.0]
PICK_POSE = [0.36043007, -0.05832796, 0.041, 1.0, 0.0, 0.0, 0.0]
ABOVE_TARGET_POSE = [0.6404910562346399,-0.0634680718364045,0.12973059202383727,0.9990895532838288,0.008019951921590617,-0.039405038638767445,0.014247379414195389]

print(f'Opening gripper and moving to pick pose {PICK_POSE}')
gripperresponse = requests.post(SERVER_URL + "open_gripper", timeout=5)
time.sleep(1) # wait for gripper to open

# Get current pose. break into position and euler angles
currpose = requests.post(SERVER_URL + "getpos", timeout=5).json()['pose']
curr_position = currpose[:3]
curr_quat = currpose[3:]
r = R.from_quat(curr_quat) # convert quat to euler
curr_euler = r.as_euler('xyz', degrees=False)

# Break above pick pose into position and euler angles
above_pick_position = ABOVE_PICK_POSE[:3]
above_pick_quat = ABOVE_PICK_POSE[3:]
r = R.from_quat(above_pick_quat) # convert quat to euler
above_pick_euler = r.as_euler('xyz', degrees=False)
# TODO: Add motion to ABOVE_PICK_POSE

# Break pick pose into position and euler angles
pick_position = PICK_POSE[:3]
pick_quat = PICK_POSE[3:]
r = R.from_quat(pick_quat) # convert quat to euler
pick_euler = r.as_euler('xyz', degrees=False)

# Break above target pose into position and euler angles
above_target_position = ABOVE_TARGET_POSE[:3]
above_target_quat = ABOVE_TARGET_POSE[3:]
r = R.from_quat(above_target_quat) # convert quat to euler
above_target_euler = r.as_euler('xyz', degrees=False)

# Lambda functions to compute the difference between poses
diffeuler = lambda a, b: (a - b + np.pi) % (2 * np.pi) - np.pi  # radians
diffpos = lambda a, b: np.array(a) - np.array(b)  # meters
is_pose_close = lambda eu1, eu2, pos1, pos2: np.allclose(diffeuler(eu1, eu2), [0, 0, 0], atol=0.1) and np.allclose(pos1, pos2, atol=0.01, rtol=0.01)

# Send move command and wait until successful
response = requests.post(SERVER_URL + "pose", json={"arr": PICK_POSE}, timeout=5) # must be list
count = 0
while not is_pose_close(curr_euler, pick_euler, curr_position, pick_position):
    currpose = requests.post(SERVER_URL + "getpos", timeout=5).json()['pose']
    curr_position = currpose[:3]
    curr_quat = currpose[3:]
    r = R.from_quat(curr_quat) # convert quat to euler
    curr_euler = r.as_euler('xyz', degrees=False)

    # print(f'Moving to goal pose {GOAL_POSE}')
    print(f'Diff euler condition {diffeuler(curr_euler, pick_euler)}')
    print(f'Diff position condition {diffpos(curr_position, pick_position)}')
    time.sleep(1)
    count += 1
    if count > 30:
        print("joint reset TIMEOUT")
        break

print('Closing gripper and moving above pick pose')
gripperresponse = requests.post(SERVER_URL + "close_gripper", timeout=5)
time.sleep(2) # wait for gripper to close (slower than gripper open)

abovepickpose = PICK_POSE.copy()
abovepickpose[2] += 0.1
response = requests.post(SERVER_URL + "pose", json={"arr": abovepickpose}, timeout=5) # must be list
time.sleep(2) # wait for gripper to close (slower than gripper open)

print(f'Moving to above target pose {ABOVE_TARGET_POSE}')

# Send move command and wait until successful
response = requests.post(SERVER_URL + "pose", json={"arr": ABOVE_TARGET_POSE}, timeout=5) # must be list
count = 0
while not is_pose_close(curr_euler, above_target_euler, curr_position, above_target_position):
    currpose = requests.post(SERVER_URL + "getpos", timeout=5).json()['pose']
    curr_position = currpose[:3]
    curr_quat = currpose[3:]
    r = R.from_quat(curr_quat) # convert quat to euler
    curr_euler = r.as_euler('xyz', degrees=False)

    # print(f'Moving to above target pose {ABOVE_TARGET_POSE}')
    print(f'Diff euler condition {diffeuler(curr_euler, above_target_euler)}')
    print(f'Diff position condition {diffpos(curr_position, above_target_position)}')
    time.sleep(1)
    count += 1
    if count > 30:
        print("joint reset TIMEOUT")
        break

print('Done!')
