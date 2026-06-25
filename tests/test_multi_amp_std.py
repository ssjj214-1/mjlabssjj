import torch

from mjlab.third_party.amp_rsl_rl.algorithms.multi_amp_ppo import MULTIAMPPPO
from mjlab.third_party.amp_rsl_rl.modules.actor_critic import ActorCritic


def test_multi_amp_ppo_clamps_scalar_policy_std_to_min_std() -> None:
  policy = ActorCritic(
    num_actor_obs=3,
    num_critic_obs=3,
    num_actions=2,
    actor_hidden_dims=(4,),
    critic_hidden_dims=(4,),
    init_noise_std=1.0,
    noise_std_type="scalar",
  )
  with torch.no_grad():
    policy.std.copy_(torch.tensor([-0.1, 0.2]))

  min_std = torch.tensor([0.05, 0.1])
  MULTIAMPPPO(policy, {}, {}, {}, min_std=min_std)

  policy.update_distribution(torch.zeros(1, 3))

  assert torch.all(policy.action_std >= min_std)
