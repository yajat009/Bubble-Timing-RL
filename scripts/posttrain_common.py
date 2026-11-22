# Shared parts of the post-training scripts: rollouts with the backbone and the head, the statistics of a rollout,
# the reference statistics of a simulation, and the rewards R2 (snapshot) and R3 (timing).
#
# All rollout statistics use the 128 x 128 grid, so a reward is fast enough to compute inside the training loop:
#   - vapor mask: a 128 x 128 cell is vapor if more than half of its 4 x 4 fine cells are vapor (dfun > 0);
#   - departures: the frozen detector with thresholds scaled to the coarse grid (detect_departures.py, cell=4);
#   - heat flux: the same wall heat-flux proxy as dwell_times_v2.py and compare_rollout_timing_v2.py.
# The final evaluation uses the full 512 x 512 detector (compare_rollout_timing_v2.py), not these coarse versions.
import json
import numpy as np, torch
from detect_departures import find_departures
from bubbleformer_io import FIELDS, load_model, fluid_params, read_frames
from fm_head import W, pool128, sample

DATA = "/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D"
FRAME_DT = 0.1           # time units between frames in BubbleML 2.0
FIRST = 105              # first frame that Bubbleformer predicts from start frames 100-104
SMOOTH = 10              # frames over which the heat flux is smoothed to label quiet and active periods


def load_meta(twall, data=DATA):
    return json.load(open(f"{data}/Twall_{twall}.json"))


def wall_mask(meta):
    """Columns of the 128 x 128 grid that lie over the heater."""
    h = meta["heater"]
    x = meta["x_min"] + (np.arange(128) + 0.5) * (meta["x_max"] - meta["x_min"]) / 128
    return torch.from_numpy((x >= h["xMin"]) & (x <= h["xMax"]))


def heat_flux(dfun_wall, temp_wall, wall_temp):
    """Wall heat-flux proxy per frame from the bottom row of the 128 x 128 fields (same formula as the other scripts).

    dfun_wall, temp_wall: (..., columns over the heater).
    """
    return (0.054 * (dfun_wall < 0) * (wall_temp - temp_wall) / (0.0007 / 32)).mean(-1)


class Backbone:
    """The frozen Bubbleformer-S model, run in chunks so that a group of rollouts fits in GPU memory."""

    def __init__(self, ckpt, dev, chunk=4):
        self.model, diff, div = load_model(ckpt, dev)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.shift = torch.tensor([diff[k] for k in FIELDS], device=dev).view(1, 1, 4, 1, 1)
        self.scale = torch.tensor([div[k] for k in FIELDS], device=dev).view(1, 1, 4, 1, 1)
        self.dev, self.chunk = dev, chunk

    @torch.no_grad()
    def __call__(self, x, fluid):
        """x: (B, 5, 4, 512, 512) in physical units; fluid: (B, 9). Returns the next 5 frames, physical units."""
        out = []
        for i in range(0, len(x), self.chunk):
            xi = (x[i:i + self.chunk] - self.shift) / self.scale
            out.append(self.model(xi, fluid[i:i + self.chunk]) * self.scale + self.shift)
        return torch.cat(out)


@torch.no_grad()
def rollout(backbone, head, scales, x, fluid, twall, meta, n_bundles, keep=0, full_vapor=False, steps=8):
    """Roll the backbone (and the head, if not None) forward n_bundles bundles from the start frames x.

    x: (B, 5, 4, 512, 512) physical units on the GPU; fluid: (B, 9); twall: (B,).
    keep: number of bundles per rollout whose head inputs and outputs are kept for the training update.
    Returns a dict with, per rollout: q (frames,), vapor128 (frames, 128, 128) bool, and if asked,
    vapor512 (frames, 512, 512) bool and fields128 (frames, 4, 128, 128) for the evaluation files.
    """
    B, dev = len(x), x.device
    cols = wall_mask(meta).to(dev)
    keep_at = set(np.random.choice(n_bundles, size=min(keep, n_bundles), replace=False).tolist()) if keep else set()
    q, vap, vap512, f128, trans = [], [], [], [], []
    for b in range(n_bundles):
        pred = backbone(x, fluid)                                            # (B, 5, 4, 512, 512)
        if head is not None:
            pred128 = pool128(pred)
            c = scales.condition(pred128, pool128(x[:, -1]), twall)
            z0, z = sample(head, c, steps)
            r = scales.to_residual(z)                                        # (B, 5, 4, 128, 128)
            pred = pred + torch.nn.functional.interpolate(
                r.flatten(0, 1), scale_factor=4, mode="bilinear", align_corners=False).view_as(pred)
            if b in keep_at:
                trans.append((z0.half().cpu(), z.half().cpu(), c.half().cpu()))
        if not torch.isfinite(pred).all():
            raise FloatingPointError(f"rollout is not finite at bundle {b}")
        x = pred
        vapor = x[:, :, 0] > 0                                               # (B, 5, 512, 512)
        p128 = pool128(x)
        q.append(heat_flux(p128[:, :, 0, 0, cols], p128[:, :, 1, 0, cols], meta["heater"]["wallTemp"]).cpu())
        vap.append((pool128(vapor.float()) > 0.5).cpu())
        if full_vapor:
            vap512.append(vapor.cpu()); f128.append(p128.half().cpu())
    out = {"q": torch.cat(q, 1).numpy(), "vapor128": torch.cat(vap, 1).numpy(), "transitions": trans}
    if full_vapor:
        out["vapor512"], out["fields128"] = torch.cat(vap512, 1).numpy(), torch.cat(f128, 1).numpy()
    return out


def dwell_times(q, threshold):
    """Lengths of the quiet runs (smoothed heat flux below threshold). Runs cut by the ends are dropped."""
    s = (np.convolve(q, np.ones(SMOOTH) / SMOOTH, mode="valid") > threshold).astype(int)
    edges = np.flatnonzero(np.diff(s)) + 1
    runs = np.split(s, edges)[1:-1]
    return np.array([len(r) for r in runs if r[0] == 0])


def stats(q, vapor128, threshold):
    """Statistics of one rollout or simulation stretch, on the 128 x 128 grid."""
    events, _ = find_departures(vapor128, FRAME_DT, cell=4)
    deps = np.array(sorted({t for t, _ in events}))
    return {"rate": 100 * len(deps) / len(q), "waits": np.diff(deps), "dwell": dwell_times(q, threshold),
            "q": q, "vapor_frac": float(vapor128.mean())}


def w1(a, b, n=100):
    """1-Wasserstein distance between two samples. An empty sample counts as far away (returns None)."""
    if len(a) < 2 or len(b) < 2:
        return None
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def distances(s, ref):
    """Distances of one rollout from the reference, for every statistic that R2 or R3 uses."""
    wait = w1(s["waits"], ref["waits"])
    dwell = w1(s["dwell"], ref["dwell"])
    return {"hist": w1(s["q"], ref["q"]),                                 # snapshot
            "vapor": abs(s["vapor_frac"] - ref["vapor_frac"]),            # snapshot
            "rate": abs(s["rate"] - ref["rate"]),                         # timing
            # Fewer than 2 departures or quiet runs: use a large distance (the longest possible wait or dwell).
            "wait": wait if wait is not None else float(len(s["q"])),     # timing
            "dwell": dwell if dwell is not None else float(len(s["q"]))}  # timing


REWARDS = {"R2": ("hist", "vapor"), "R3": ("rate", "wait", "dwell"), "R4": ("hist", "vapor", "rate", "wait", "dwell")}


def reward(kind, d, noise):
    """Reward = minus the sum of the distances, each divided by its noise scale. Higher is better."""
    return -sum(d[k] / noise[k] for k in REWARDS[kind])


def reference(twall, n_frames, data=DATA, n_noise=20, seed=0):
    """Reference statistics of one simulation, and the noise scale of each distance for rollouts of n_frames.

    The noise scale is the median distance between a random n_frames stretch of the simulation and the whole
    simulation: the distance that a perfect emulator would get by chance.
    """
    meta = load_meta(twall, data)
    cols = wall_mask(meta).numpy()
    f = np.load(f"{data}/Twall_{twall}_fields128.npy", mmap_mode="r")[FIRST:]
    v = np.load(f"{data}/Twall_{twall}_vapor.npy", mmap_mode="r")
    q = heat_flux(f[:, 0, 0][:, cols].astype(np.float32), f[:, 1, 0][:, cols].astype(np.float32), meta["heater"]["wallTemp"])
    vapor128 = np.stack([v[t].reshape(128, 4, 128, 4).mean(axis=(1, 3)) > 0.5 for t in range(FIRST, len(v))])
    threshold = float(np.median(np.convolve(q, np.ones(SMOOTH) / SMOOTH, mode="valid")))
    ref = stats(q, vapor128, threshold)
    rng = np.random.default_rng(seed)
    noise = {k: [] for k in ("hist", "vapor", "rate", "wait", "dwell")}
    for _ in range(n_noise):
        s0 = rng.integers(0, len(q) - n_frames)
        d = distances(stats(q[s0:s0 + n_frames], vapor128[s0:s0 + n_frames], threshold), ref)
        for k in noise:
            noise[k].append(d[k])
    # Floor: 5% of the typical size of each quantity, so that no term gets a near-zero scale and dominates the reward.
    typical = {"hist": float(np.std(ref["q"])), "vapor": ref["vapor_frac"], "rate": ref["rate"],
               "wait": float(np.mean(ref["waits"])) if len(ref["waits"]) else 1.0,
               "dwell": float(np.mean(ref["dwell"])) if len(ref["dwell"]) else 1.0}
    noise = {k: max(float(np.median(v)), 0.05 * typical[k], 1e-6) for k, v in noise.items()}
    return {"stats": ref, "noise": noise, "threshold": threshold, "meta": meta}
