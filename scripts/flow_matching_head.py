# Flow-matching head for the frozen Bubbleformer-S backbone.
#
# What it does: the backbone predicts the next 5 frames, but it always gives the same answer for the same input.
# Real boiling is partly random. The head adds a random correction (the "residual") to the backbone prediction,
# so that repeated rollouts from the same start can differ, as real boiling does.
#
# Grid: the head works on a 128 x 128 grid. Each coarse cell is the mean of 4 x 4 cells of the 512 x 512 grid.
# This keeps the head small and fast. The rollout code enlarges the residual back to 512 x 512 before it adds it.
#
# Flow matching, in short:
#   - The head is a velocity field v(z, tau, condition), where tau goes from 0 to 1.
#   - To sample a residual, start from random noise z at tau = 0 and move z along v in small steps until tau = 1.
#   - To train, pair a noise sample with a true residual. At a random tau, the point between them is
#     z_tau = (1 - tau) * noise + tau * residual, and the head must predict the direction residual - noise.
import math
import torch, torch.nn as nn, torch.nn.functional as F

FRAMES_PER_BUNDLE = 5   # the backbone reads and predicts 5 frames at a time
N_FIELDS = 4            # dfun (signed distance to the interface), temperature, x velocity, y velocity
RESIDUAL_CHANNELS = FRAMES_PER_BUNDLE * N_FIELDS                 # 20: one channel per frame and field
CONDITION_CHANNELS = FRAMES_PER_BUNDLE * N_FIELDS + N_FIELDS + 1  # 25: backbone prediction, last input frame, wall temperature


def average_to_128(fields):
    """Average each 4 x 4 block of cells: (..., 512, 512) -> (..., 128, 128)."""
    *leading, height, width = fields.shape
    return fields.reshape(*leading, height // 4, 4, width // 4, 4).mean(dim=(-3, -1))


class ConvBlock(nn.Module):
    """Two 3 x 3 convolutions with group normalization. The time embedding shifts the features between them."""

    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1, self.norm2 = nn.GroupNorm(8, out_channels), nn.GroupNorm(8, out_channels)
        self.time_shift = nn.Linear(time_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_features):
        h = F.silu(self.norm1(self.conv1(x))) + self.time_shift(time_features)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(h))) + self.skip(x)


class ResidualHead(nn.Module):
    """A small U-Net on the 128 x 128 grid.

    Input: the current noisy residual z, the flow time tau, and the condition. Output: the velocity for z.
    The U-Net goes down to 16 x 16 in 3 steps and back up, with skip connections at each size.
    """

    def __init__(self, width=64):
        super().__init__()
        time_dim = 4 * width
        self.width = width
        self.time_mlp = nn.Sequential(nn.Linear(width, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.input_conv = nn.Conv2d(RESIDUAL_CHANNELS + CONDITION_CHANNELS, width, 3, padding=1)
        self.down_128 = ConvBlock(width, width, time_dim)
        self.down_64 = ConvBlock(width, 2 * width, time_dim)
        self.down_32 = ConvBlock(2 * width, 4 * width, time_dim)
        self.middle_16 = ConvBlock(4 * width, 4 * width, time_dim)
        self.up_32 = ConvBlock(8 * width, 2 * width, time_dim)
        self.up_64 = ConvBlock(4 * width, width, time_dim)
        self.up_128 = ConvBlock(2 * width, width, time_dim)
        self.output_conv = nn.Conv2d(width, RESIDUAL_CHANNELS, 3, padding=1)
        # Start with a zero output: before training, the head adds no correction.
        nn.init.zeros_(self.output_conv.weight); nn.init.zeros_(self.output_conv.bias)

    def time_features(self, tau):
        """Sine and cosine features of tau (one value per sample), passed through a small MLP."""
        half = self.width // 2
        frequencies = torch.exp(-math.log(1000.0) * torch.arange(half, device=tau.device) / half)
        angles = 1000.0 * tau[:, None] * frequencies[None]
        return self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=1))

    def forward(self, z, tau, condition):
        t = self.time_features(tau)
        h128 = self.down_128(self.input_conv(torch.cat([z, condition], dim=1)), t)
        h64 = self.down_64(F.avg_pool2d(h128, 2), t)
        h32 = self.down_32(F.avg_pool2d(h64, 2), t)
        h = self.middle_16(F.avg_pool2d(h32, 2), t)
        h = self.up_32(torch.cat([F.interpolate(h, scale_factor=2), h32], dim=1), t)
        h = self.up_64(torch.cat([F.interpolate(h, scale_factor=2), h64], dim=1), t)
        h = self.up_128(torch.cat([F.interpolate(h, scale_factor=2), h128], dim=1), t)
        return self.output_conv(h)


class Normalizer:
    """Converts between physical units and the normalized values that the head sees.

    It holds, per field, the mean and standard deviation of the simulation fields and the standard deviation of the
    residuals (true frames minus backbone prediction). prepare_head_data_v2.py computes them from the training files.
    """

    def __init__(self, values):
        self.values = values   # the dict that is saved with each head checkpoint
        self.field_mean = torch.tensor(values["field_mean"]).view(1, 1, N_FIELDS, 1, 1)
        self.field_std = torch.tensor(values["field_std"]).view(1, 1, N_FIELDS, 1, 1)
        self.residual_std = torch.tensor(values["residual_std"]).view(1, 1, N_FIELDS, 1, 1)
        self.wall_temp_mean, self.wall_temp_std = values["twall_mean"], values["twall_std"]

    def to(self, device):
        for name in ("field_mean", "field_std", "residual_std"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def build_condition(self, prediction_128, last_input_128, wall_temp):
        """Stack the inputs of the head into one tensor of CONDITION_CHANNELS channels.

        prediction_128: backbone prediction, (B, 5, 4, 128, 128); last_input_128: last input frame, (B, 4, 128, 128);
        wall_temp: heater wall temperature in degrees C, (B,). Returns (B, 25, 128, 128).
        """
        prediction = ((prediction_128 - self.field_mean) / self.field_std).flatten(1, 2)
        last_input = ((last_input_128[:, None] - self.field_mean) / self.field_std)[:, 0]
        wall = ((wall_temp - self.wall_temp_mean) / self.wall_temp_std).view(-1, 1, 1, 1)
        wall = wall.expand(-1, 1, *prediction.shape[-2:])
        return torch.cat([prediction, last_input, wall], dim=1)

    def to_physical_residual(self, normalized):
        """(B, 20, H, W) normalized residual -> (B, 5, 4, H, W) residual in physical units."""
        return normalized.view(normalized.shape[0], FRAMES_PER_BUNDLE, N_FIELDS, *normalized.shape[-2:]) * self.residual_std

    def to_normalized_residual(self, physical):
        """(B, 5, 4, H, W) residual in physical units -> (B, 20, H, W) normalized residual."""
        return (physical / self.residual_std).flatten(1, 2)


@torch.no_grad()
def sample_residual(head, condition, steps=8, noise=None):
    """Draw one normalized residual per condition by moving random noise along the head's velocity in Euler steps.

    Returns (noise, residual). The rollout code keeps both, because the post-training update needs them.
    """
    if noise is None:
        noise = torch.randn(condition.shape[0], RESIDUAL_CHANNELS, *condition.shape[-2:], device=condition.device)
    z = noise
    for k in range(steps):
        tau = torch.full((condition.shape[0],), k / steps, device=condition.device)
        z = z + head(z, tau, condition) / steps
    return noise, z


def flow_matching_loss(head, residual, condition):
    """Flow-matching loss for normalized residuals (B, 20, H, W) and their conditions."""
    noise = torch.randn_like(residual)
    tau = torch.rand(residual.shape[0], device=residual.device)
    tau_4d = tau.view(-1, 1, 1, 1)
    z_tau = (1 - tau_4d) * noise + tau_4d * residual
    return ((head(z_tau, tau, condition) - (residual - noise)) ** 2).mean()
