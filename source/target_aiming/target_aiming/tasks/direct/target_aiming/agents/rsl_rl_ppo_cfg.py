# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlCNNModelCfg, RslRlMLPModelCfg, RslRlPpoAlgorithmCfg


# Shared CNN configuration for the image encoder
_CNN_CFG = RslRlCNNModelCfg.CNNCfg(
    output_channels=[32, 64, 128],
    kernel_size=[8, 4, 3],
    stride=[4, 2, 2],
    activation="elu",
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 10000
    save_interval = 50
    experiment_name = "target_aiming_direct"

    # Map observation groups to actor and critic
    obs_groups = {
        "actor": ["image", "state"],
        "critic": ["image", "state"],
    }

    actor = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=True,
        init_noise_std=0.3,
        cnn_cfg=_CNN_CFG,
    )

    critic = RslRlCNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=False,
        cnn_cfg=_CNN_CFG,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        share_cnn_encoders=True,
    )
