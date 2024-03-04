import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import optax
from tqdm import tqdm
import gymnasium as gym
import datetime

from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.utils.launcher import make_wandb_logger

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    Quat2EulerWrapper,
)
from serl_launcher.networks.reward_classifier import create_classifier

from absl import app, flags, logging

import franka_env
from franka_env.envs.relative_env import RelativeFrame

FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", True, "Debug disables wandb logging")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_string("env", "FrankaPCBInsert-Vision-v0", "Name of environment.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("data_file", "peg_insert_5_demos_2024-02-23_11-15-06.pkl.pkl", "Path to data file.")

def main(_):
    num_epochs = 100
    batch_size = 256

    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    devices = jax.local_devices()
    num_devices = len(devices)
    sharding = jax.sharding.PositionalSharding(devices)

    wandb_logger = make_wandb_logger(
            project="serl_dev",
            description=FLAGS.exp_name or FLAGS.env,
            debug=FLAGS.debug,
        )

    env = gym.make("FrankaPCBInsert-Vision-v0", fake_env=True, save_video=False)
    env = GripperCloseEnv(env)
    env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
    image_keys = [k for k in env.observation_space.keys() if "state" not in k]

    pos_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=5000,
        image_keys=image_keys,
    )

    neg_buffer = MemoryEfficientReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=10000,
        image_keys=image_keys,
    )

    def add_data_to_buffer(demos_path, neg_only=False):
        """Helper function to add data to buffer"""
        demos = pkl.load(open(demos_path, "rb"))
        # add failed data to negative buffer
        failed_data = [d for d in demos if not d["rewards"]]
        for traj in failed_data:
            neg_buffer.insert(traj)

        # add success data to positive buffer
        if not neg_only:
            success_data = [d for d in demos if d["rewards"]]
            for traj in success_data:
                pos_buffer.insert(traj)

    add_data_to_buffer(FLAGS.data_file)
    # add_data_to_buffer("cable_route_20_failed_2024-01-04_12-50-24.pkl", neg_only=True)

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")
    pos_fraction = 0.5  # for re-balancing the dataset
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": int(batch_size * pos_fraction),
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": int(batch_size * (1 - pos_fraction)),
            "pack_obs_and_next_obs": False,
        },
        device=sharding.replicate(),
    )

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
    classifier = create_classifier(key, sample["next_observations"], image_keys)

    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    # Define the training step
    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["data"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["data"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    # Training Loop
    for epoch in tqdm(range(num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        sample = concat_batches(
            pos_sample["observations"], neg_sample["observations"], axis=0
        )   
        rng, key = jax.random.split(rng)
        sample = data_augmentation_fn(key, sample)
        labels = jnp.concatenate(
            [jnp.ones((int(batch_size * pos_fraction), 1)), jnp.zeros((int(batch_size * (1 - pos_fraction)), 1))], axis=0
        )
        batch = {"data": sample, "labels": labels}

        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)
        if wandb_logger:
            wandb_logger.log({"Train Loss": train_loss}, step=epoch+1)
            wandb_logger.log({"Train Accuracy": train_accuracy}, step=epoch+1)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    now = datetime.datetime.now()
    checkpoints.save_checkpoint(
        f"/home/gautamsalhotra/serl/examples/async_pcb_insert_drq/classifier_ckpts_{now.strftime('%m.%d.%H.%M')}",
        classifier,
        step=num_epochs,
        overwrite=True,
    )

if __name__ == "__main__":
    app.run(main)
