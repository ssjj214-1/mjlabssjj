"""RL configuration for the Ultra GameYaw velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def ultra_game_yaw_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner configuration matching the Isaac Lab Ultra agent cfg.

  Network sizes and PPO hyper-parameters are taken from
  `UltraGameYawMotionRunAgentCfg`. Phase-1 omits AMP/HIM/symmetry — those
  are layered in via custom runner classes in Phase-2.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="ultra_game_yaw_velocity",
    save_interval=50,
    num_steps_per_env=32,  # Match Isaac cfg.
    max_iterations=20_000,
  )
