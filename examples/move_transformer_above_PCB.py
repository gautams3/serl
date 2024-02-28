""" Scipt to move the transformer above the PCB. """

import os
import requests
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
np.set_printoptions(precision=3, suppress=True)
SERVER_URL = "http://127.0.0.1:5000/"
from serl_robot_infra.franka_env.envs.pcb_env.config import PCBEnvConfig

# Fixed pick pose for hopper
ABOVE_PICK_POSE = [0.36043007, -0.05832796, 0.101, 1.0, 0.0, 0.0, 0.0]
PICK_POSE = [0.36043007, -0.05832796, 0.041, 1.0, 0.0, 0.0, 0.0]
ABOVE_TARGET_POSE = [0.6404910562346399,-0.0634680718364045,0.12973059202383727,0.9990895532838288,0.008019951921590617,-0.039405038638767445,0.014247379414195389]

requests.post(SERVER_URL + "update_param", json=PCBEnvConfig.PRECISION_PARAM)
time.sleep(0.5)

print(f'Opening gripper and moving to pick pose {PICK_POSE}')
gripperresponse = requests.post(SERVER_URL + "open_gripper", timeout=5)
time.sleep(1) # wait for gripper to open

# Get current pose. break into position and euler angles
currpose = requests.post(SERVER_URL + "getpos", timeout=5).json()['pose']
curr_position = currpose[:3]
curr_quat = currpose[3:]
r = R.from_quat(curr_quat) # convert quat to euler
curr_euler = r.as_euler('xyz', degrees=False)

# Break pick pose into position and euler angles
pick_position = PICK_POSE[:3]
pick_quat = PICK_POSE[3:]
r = R.from_quat(pick_quat) # convert quat to euler
pick_euler = r.as_euler('xyz', degrees=False)

abovepickpose = PICK_POSE.copy()
abovepickpose[2] += 0.1

# Break above target pose into position and euler angles
above_target_position = ABOVE_TARGET_POSE[:3]
above_target_quat = ABOVE_TARGET_POSE[3:]
r = R.from_quat(above_target_quat) # convert quat to euler
above_target_euler = r.as_euler('xyz', degrees=False)

# Lambda functions to compute the difference between poses
diffeuler = lambda a, b: (a - b + np.pi) % (2 * np.pi) - np.pi  # radians
diffpos = lambda a, b: np.array(a) - np.array(b)  # meters
is_pose_close = lambda eu1, eu2, pos1, pos2: np.allclose(diffeuler(eu1, eu2), [0, 0, 0], atol=0.1) and np.allclose(pos1, pos2, atol=0.01, rtol=0.01)

# Move above pick pose
response = requests.post(SERVER_URL + "pose", json={"arr": ABOVE_PICK_POSE}, timeout=5) # must be list
time.sleep(2)  # HACK: wait for robot to move

# move to pick pose and wait until successful
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

response = requests.post(SERVER_URL + "pose", json={"arr": abovepickpose}, timeout=5) # must be list
input('Press enter to continue')

print(f'Moving to above target pose {ABOVE_TARGET_POSE}')

# Send move command and wait until successful
response = requests.post(SERVER_URL + "pose", json={"arr": ABOVE_TARGET_POSE}, timeout=5) # must be list
input('Press enter to continue')

### Alternative: Move to a joint pose instead of a cartesian pose
# print('waiting before final move')
# time.sleep(2)

# response = input('Kill old controller first. Then start roscore and press enter')
# if response == 'q':
#     print('Exiting...')
#     exit()

# above_pick_q = [0.1756412070337813, 0.03122043978383666, -0.3287547466964052, -2.8399477963528117, 0.06180817775113116, 2.855340080923504, 0.5120437213759724]
# print('Setting joint goal...')
# above_target_q = [0.19097669621114738, 0.6011634117511281, -0.30170746777767066, -1.8931544224643337, 0.26372237982352575, 2.5055904395050472, 0.5184927675111416]
# setJointGoal = f"rosparam set /target_joint_positions '{above_target_q}' #ABOVE_TARGET_Q"
# os.system(setJointGoal)

# print('Starting joint controller. Make sure to send SIGINT after the joint goal is reached. Press enter')
# sendGoal = "roslaunch serl_franka_controllers joint.launch robot_ip:=192.170.10.4 load_gripper:=true"
# os.system(sendGoal)

# print('Now start the other controller and press enter')

print(f'Setting params to PCBEnvConfig.COMPLIANCE_PARAM')
requests.post(SERVER_URL + "update_param", json=PCBEnvConfig.COMPLIANCE_PARAM)
time.sleep(0.5)

print('Done!')
