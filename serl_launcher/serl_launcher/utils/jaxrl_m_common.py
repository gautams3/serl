# !/usr/bin/env python3

# NOTE: this requires jaxrl_m to be installed:
#       https://github.com/rail-berkeley/jaxrl_minimal

from agentlace.trainer import TrainerConfig

import jax
from jax import nn
from jaxrl_m.agents.continuous.sac import SACAgent
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.agents.continuous.drq import DrQAgent
from jaxrl_m.vision.small_encoders import SmallEncoder
from jaxrl_m.vision.mobilenet import MobileNetEncoder
from jaxrl_m.vision.resnet_v1 import resnetv1_configs


##############################################################################


def make_sac_agent(seed, sample_obs, sample_action):
    return SACAgent.create_states(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=0.99,
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=2,
    )


def make_drq_agent(
    seed, sample_obs, sample_action, image_keys=("image",), encoder_type="small"
):
    agent = DrQAgent.create_drq(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=0.96,  # 0.99
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=2,
    )
    return agent


def make_trainer_config():
    return TrainerConfig(
        port_number=5488, broadcast_port=5489, request_types=["send-stats"]
    )


def make_wandb_logger(
    project: str = "agentlace",
    description: str = "jaxrl_m",
    debug: bool = False,
):
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": project,
            "exp_descriptor": description,
            "tag": description,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant={},
        debug=debug,
    )
    return wandb_logger