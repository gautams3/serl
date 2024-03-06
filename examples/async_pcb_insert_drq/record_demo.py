import gymnasium as gym
from tqdm import tqdm
import jax
import numpy as np
from absl import app, flags
import copy
import pickle as pkl
import datetime
import os

import franka_env

from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
    Quat2EulerWrapper,
    BinaryRewardClassifierWrapper,
)

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from serl_launcher.wrappers.chunking import ChunkingWrapper

flags.DEFINE_boolean("no_truncate", False, "Whether to truncate the episode on done")
flags.DEFINE_integer("success_needed", 20, "Number of successful demonstrations to record")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_string(
    "reward_classifier_ckpt_path", None, "Path to reward classifier ckpt."
)
FLAGS = flags.FLAGS

def main(_):
    """Record a set of demonstrations for the PCB insertion task."""
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    env = gym.make("FrankaPCBInsert-Vision-v0", save_video=False)
    env = GripperCloseEnv(env)
    env = SpacemouseIntervention(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    if FLAGS.reward_classifier_ckpt_path is not None:
        print(f'Using reward classifier from {FLAGS.reward_classifier_ckpt_path}')

        image_keys = [key for key in env.observation_space.keys() if key != "state"]
        # initialize the classifier, if specified, and wrap the env
        from serl_launcher.networks.reward_classifier import load_classifier_func

        # if FLAGS.reward_classifier_ckpt_path is None:
        #     raise ValueError("reward_classifier_ckpt_path must be specified for actor")

        reward_func = load_classifier_func(
            key=sampling_rng,
            sample=env.observation_space.sample(),
            image_keys=image_keys,
            checkpoint_path=FLAGS.reward_classifier_ckpt_path,
            step=100,
        )
        env = BinaryRewardClassifierWrapper(env, reward_func)
    else:
        print("No reward classifier specified. Using environment reward.")

    obs, _ = env.reset()

    transitions = []
    success_count = 0
    success_needed = FLAGS.success_needed
    total_count = 0
    pbar = tqdm(total=success_needed)

    uuid = datetime.datetime.now().strftime("%m.%d.%H.%M")
    file_name = f"pcb_insert_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(file_path):
        raise FileExistsError(f"{file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    while success_count < success_needed:
        next_obs, rew, done, truncated, info = env.step(action=np.zeros((6,)))
        actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        transitions.append(transition)

        obs = next_obs

        should_reset = done
        if FLAGS.no_truncate:  # only reset if max length reached. records many more success transitions
            should_reset = env.curr_path_length >= env.max_episode_length

        if should_reset:
            success_count += rew
            total_count += 1
            print(
                f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
            )
            pbar.update(rew)
            obs, _ = env.reset()

    with open(file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_path}")

    env.close()
    pbar.close()
    print("Done.")

if __name__ == "__main__":
    app.run(main)
