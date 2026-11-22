# Flow-matching head for the frozen Bubbleformer-S backbone.
#
# The backbone gives one deterministic prediction of the next 5 frames. Real boiling is partly random, so the head
# adds a random correction (the residual) to that prediction. The head works on a 128 x 128 grid (each cell is the
# mean of 4 x 4 cells of the 512 x 512 grid), which keeps it small and fast. The residual is enlarged back to
# 512 x 512 before it is added to the backbone prediction.
#
# Flow matching: the head is a velocity field v(z, tau, c). Sampling starts from noise z0 ~ N(0, I) at tau = 0 and
# follows v with Euler steps to tau = 1, where z is a residual sample. Training pairs a noise sample z0 with a true
# residual x and fits v(z_tau, tau, c) to x - z0 at z_tau = (1 - tau) z0 + tau x.
import math
import torch, torch.nn as nn, torch.nn.functional as F

W, NF = 5, 4             # frames per bundle, fields per frame
X_CH = W * NF            # residual channels (5 frames x 4 fields)
C_CH = W * NF + NF + 1   # condition channels: backbone prediction, last input frame, wall temperature


def pool128(a):
    """Average each 4 x 4 block of cells: (..., 512, 512) -> (..., 128, 128)."""
    *lead, h, w = a.shape
    return a.reshape(*lead, h // 4, 4, w // 4, 4).mean(dim=(-3, -1))


class Block(nn.Module):
    """Two 3 x 3 convolutions with group norm. The time embedding shifts the features between them."""

    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(c_in, c_out, 3, padding=1), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm1, self.norm2 = nn.GroupNorm(8, c_out), nn.GroupNorm(8, c_out)
        self.t = nn.Linear(t_dim, c_out)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, t):
        h = F.silu(self.norm1(self.conv1(x))) + self.t(t)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(h))) + self.skip(x)


class Head(nn.Module):
    """A small U-Net on the 128 x 128 grid: input (residual sample, condition), output velocity."""

    def __init__(self, width=64):
        super().__init__()
        t_dim = 4 * width
        self.width = width
        self.t_mlp = nn.Sequential(nn.Linear(width, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.inp = nn.Conv2d(X_CH + C_CH, width, 3, padding=1)
        self.d1, self.d2, self.d3 = Block(width, width, t_dim), Block(width, 2 * width, t_dim), Block(2 * width, 4 * width, t_dim)
        self.mid = Block(4 * width, 4 * width, t_dim)
        self.u3, self.u2, self.u1 = Block(8 * width, 2 * width, t_dim), Block(4 * width, width, t_dim), Block(2 * width, width, t_dim)
        self.out = nn.Conv2d(width, X_CH, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def time_embedding(self, tau):
        half = self.width // 2
        freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=tau.device) / half)
        angles = 1000.0 * tau[:, None] * freqs[None]
        return self.t_mlp(torch.cat([angles.sin(), angles.cos()], dim=1))

    def forward(self, z, tau, c):
        t = self.time_embedding(tau)
        h1 = self.d1(self.inp(torch.cat([z, c], dim=1)), t)          # 128 x 128
        h2 = self.d2(F.avg_pool2d(h1, 2), t)                           # 64 x 64
        h3 = self.d3(F.avg_pool2d(h2, 2), t)                           # 32 x 32
        h = self.mid(F.avg_pool2d(h3, 2), t)                           # 16 x 16
        h = self.u3(torch.cat([F.interpolate(h, scale_factor=2), h3], 1), t)
        h = self.u2(torch.cat([F.interpolate(h, scale_factor=2), h2], 1), t)
        h = self.u1(torch.cat([F.interpolate(h, scale_factor=2), h1], 1), t)
        return self.out(h)


class Scales:
    """Per-field mean and standard deviation of the fields and of the residuals, from the training files.

    The head sees normalized values. These numbers map between normalized and physical units.
    """

    def __init__(self, d):
        self.d = d
        self.field_mean = torch.tensor(d["field_mean"]).view(1, 1, NF, 1, 1)
        self.field_std = torch.tensor(d["field_std"]).view(1, 1, NF, 1, 1)
        self.res_std = torch.tensor(d["residual_std"]).view(1, 1, NF, 1, 1)
        self.twall_mean, self.twall_std = d["twall_mean"], d["twall_std"]

    def to(self, dev):
        for k in ("field_mean", "field_std", "res_std"):
            setattr(self, k, getattr(self, k).to(dev))
        return self

    def condition(self, pred128, last128, twall):
        """pred128: (B, 5, 4, 128, 128); last128: (B, 4, 128, 128); twall: (B,). Returns (B, C_CH, 128, 128)."""
        p = ((pred128 - self.field_mean) / self.field_std).flatten(1, 2)
        l = ((last128[:, None] - self.field_mean) / self.field_std)[:, 0]
        t = ((twall - self.twall_mean) / self.twall_std).view(-1, 1, 1, 1).expand(-1, 1, *p.shape[-2:])
        return torch.cat([p, l, t], dim=1)

    def to_residual(self, x):
        """Normalized residual (B, 20, H, W) -> physical residual (B, 5, 4, H, W)."""
        return x.view(x.shape[0], W, NF, *x.shape[-2:]) * self.res_std

    def from_residual(self, r):
        """Physical residual (B, 5, 4, H, W) -> normalized residual (B, 20, H, W)."""
        return (r / self.res_std).flatten(1, 2)


@torch.no_grad()
def sample(head, c, steps=8, z0=None):
    """Draw one normalized residual per condition with Euler steps. Returns (z0, sample)."""
    if z0 is None:
        z0 = torch.randn(c.shape[0], X_CH, *c.shape[-2:], device=c.device)
    z = z0
    for k in range(steps):
        tau = torch.full((c.shape[0],), k / steps, device=c.device)
        z = z + head(z, tau, c) / steps
    return z0, z


def cfm_loss(head, x, c):
    """Conditional flow-matching loss for normalized residuals x (B, 20, H, W) and conditions c."""
    z0 = torch.randn_like(x)
    tau = torch.rand(x.shape[0], device=x.device)
    z = (1 - tau.view(-1, 1, 1, 1)) * z0 + tau.view(-1, 1, 1, 1) * x
    return ((head(z, tau, c) - (x - z0)) ** 2).mean()
