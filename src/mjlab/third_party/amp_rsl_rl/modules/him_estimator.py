import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class HIMGRUEncoder(nn.Module):
  """Encode a fixed obs-history window with a GRU while keeping the flat input API."""

  def __init__(
    self,
    temporal_steps,
    num_one_step_obs,
    output_dim,
    hidden_dim=128,
    num_layers=1,
    head_hidden_dims=(64,),
    activation="elu",
  ):
    super().__init__()
    self.temporal_steps = int(temporal_steps)
    self.num_one_step_obs = int(num_one_step_obs)
    self.rnn = nn.GRU(
      input_size=self.num_one_step_obs,
      hidden_size=int(hidden_dim),
      num_layers=int(num_layers),
      batch_first=True,
    )
    act = get_activation(activation)
    head_layers = []
    head_input_dim = int(hidden_dim)
    for layer_dim in head_hidden_dims:
      head_layers += [nn.Linear(head_input_dim, int(layer_dim)), act]
      head_input_dim = int(layer_dim)
    head_layers += [nn.Linear(head_input_dim, int(output_dim))]
    self.head = nn.Sequential(*head_layers)

  def forward(self, obs_history):
    obs_seq = obs_history.reshape(-1, self.temporal_steps, self.num_one_step_obs)
    _, hidden = self.rnn(obs_seq)
    return self.head(hidden[-1])


class HIMEstimator(nn.Module):
  """History-based internal model: estimate yaw-frame lin vel + latent from actor obs history."""

  def __init__(
    self,
    temporal_steps,
    num_one_step_obs,
    num_one_step_critic_obs,
    vel_index_in_critic,
    vel_scale_in_critic: float = 1.0,
    enc_hidden_dims=(128, 64, 32),
    tar_hidden_dims=(128, 64),
    activation="elu",
    learning_rate=1e-3,
    max_grad_norm=10.0,
    num_prototype=32,
    temperature=3.0,
    encoder_type="mlp",
    gru_hidden_dim=128,
    gru_num_layers=1,
    gru_head_hidden_dims=(64,),
    **kwargs,
  ):
    if kwargs:
      print(
        "HIMEstimator.__init__ got unexpected arguments, which will be ignored: "
        + str([key for key in kwargs.keys()])
      )
    super().__init__()
    activation_name = activation
    activation = get_activation(activation)

    self.temporal_steps = int(temporal_steps)
    self.num_one_step_obs = int(num_one_step_obs)
    self.num_one_step_critic_obs = int(num_one_step_critic_obs)
    self.vel_index_in_critic = int(vel_index_in_critic)
    self.vel_scale_in_critic = float(vel_scale_in_critic)
    self.num_latent = int(enc_hidden_dims[-1])
    self.max_grad_norm = max_grad_norm
    self.temperature = temperature
    self.encoder_type = str(encoder_type).lower()

    if self.encoder_type == "gru":
      self.encoder = HIMGRUEncoder(
        temporal_steps=self.temporal_steps,
        num_one_step_obs=self.num_one_step_obs,
        output_dim=self.num_latent + 3,
        hidden_dim=gru_hidden_dim,
        num_layers=gru_num_layers,
        head_hidden_dims=gru_head_hidden_dims,
        activation=activation_name,
      )
    elif self.encoder_type == "mlp":
      enc_input_dim = self.temporal_steps * self.num_one_step_obs
      enc_layers = []
      for layer_dim in enc_hidden_dims[:-1]:
        enc_layers += [nn.Linear(enc_input_dim, layer_dim), activation]
        enc_input_dim = layer_dim
      enc_layers += [nn.Linear(enc_input_dim, self.num_latent + 3)]
      self.encoder = nn.Sequential(*enc_layers)
    else:
      raise ValueError(
        f"Unsupported HIM encoder_type '{encoder_type}'. Expected 'mlp' or 'gru'."
      )

    tar_input_dim = self.num_one_step_critic_obs
    tar_layers = []
    for layer_dim in tar_hidden_dims:
      tar_layers += [nn.Linear(tar_input_dim, layer_dim), activation]
      tar_input_dim = layer_dim
    tar_layers += [nn.Linear(tar_input_dim, self.num_latent)]
    self.target = nn.Sequential(*tar_layers)

    self.proto = nn.Embedding(num_prototype, self.num_latent)
    self.learning_rate = learning_rate
    self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

  def forward(self, obs_history):
    vel, z = self.encode(obs_history)
    return vel.detach(), z.detach()

  def encode(self, obs_history):
    parts = self.encoder(obs_history.detach())
    vel, z = parts[..., :3], parts[..., 3:]
    z = F.normalize(z, dim=-1, p=2)
    return vel, z

  def update(self, obs_history, next_critic_obs, lr=None):
    if lr is not None:
      self.learning_rate = lr
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = self.learning_rate

    # Skip the update if the inputs are non-finite. On rough terrain a transient
    # extreme physics state can make a privileged critic obs (or the velocity
    # supervision target) NaN/Inf. The estimator runs on its own optimizer, so
    # without this guard a single bad batch steps on a NaN gradient and
    # permanently poisons every estimator weight (encoder + target + proto) --
    # the same failure the PPO optimizer already guards against via
    # ``isfinite(grad_norm)``. Returning early leaves the weights untouched.
    if not (
      torch.isfinite(obs_history).all() and torch.isfinite(next_critic_obs).all()
    ):
      self.optimizer.zero_grad(set_to_none=True)
      return 0.0, 0.0

    vi = self.vel_index_in_critic
    # Critic stores yaw-frame linear velocity with env obs scaling. Convert back to m/s for supervision.
    vel = next_critic_obs[:, vi : vi + 3].detach()
    if abs(self.vel_scale_in_critic) > 1e-9:
      vel = vel / self.vel_scale_in_critic
    # Target network can consume privileged critic extras during training.
    # This does not affect deploy since only encoder + actor are exported.
    next_obs = next_critic_obs[:, : self.num_one_step_critic_obs].detach()

    z_s = self.encoder(obs_history)
    z_t = self.target(next_obs)
    pred_vel, z_s = z_s[..., :3], z_s[..., 3:]

    z_s = F.normalize(z_s, dim=-1, p=2)
    z_t = F.normalize(z_t, dim=-1, p=2)

    with torch.no_grad():
      w = F.normalize(self.proto.weight.data.clone(), dim=-1, p=2)
      self.proto.weight.copy_(w)

    score_s = z_s @ self.proto.weight.T
    score_t = z_t @ self.proto.weight.T

    with torch.no_grad():
      q_s = sinkhorn(score_s)
      q_t = sinkhorn(score_t)

    log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
    log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

    swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
    estimation_loss = F.mse_loss(pred_vel, vel)
    losses = estimation_loss + swap_loss

    self.optimizer.zero_grad()
    losses.backward()
    grad_norm = nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
    # Only step on a finite loss + gradient. A single non-finite gradient would
    # otherwise corrupt every estimator weight, and NaN weights then produce NaN
    # forever (unrecoverable). Mirrors the PPO optimizer's finite-grad guard.
    if torch.isfinite(losses) and torch.isfinite(grad_norm):
      self.optimizer.step()
    else:
      self.optimizer.zero_grad(set_to_none=True)

    return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
  """Balanced assignment (SwAV-style) on prototype scores.

  Args:
      out: (B, K) scores.
  Returns:
      (B, K) soft assignment, each row sums to 1.
  """
  # Subtract the max before exp (standard SwAV stabilization) so a large score
  # can't overflow to +Inf. Denominators are floored with a tiny epsilon so an
  # underflowed row/col sum can't produce Inf/NaN.
  out = out - out.max()
  Q = torch.exp(out / eps).T  # (K, B)
  k, b = Q.shape[0], Q.shape[1]
  Q /= Q.sum().clamp_min(1e-12)

  for _ in range(iters):
    # normalize rows: total weight per prototype must be 1/K
    Q /= torch.sum(Q, dim=1, keepdim=True).clamp_min(1e-12)
    Q /= k
    # normalize cols: total weight per sample must be 1/B
    Q /= torch.sum(Q, dim=0, keepdim=True).clamp_min(1e-12)
    Q /= b
  return (Q * b).T


def get_activation(act_name):
  if act_name == "elu":
    return nn.ELU()
  if act_name == "selu":
    return nn.SELU()
  if act_name == "relu":
    return nn.ReLU()
  if act_name == "silu":
    return nn.SiLU()
  if act_name == "lrelu":
    return nn.LeakyReLU()
  if act_name == "tanh":
    return nn.Tanh()
  if act_name == "sigmoid":
    return nn.Sigmoid()
  raise ValueError(f"invalid activation function: {act_name}")
