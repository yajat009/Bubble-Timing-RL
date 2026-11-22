# Shared parts of the post-training scripts:
#   - Backbone: the frozen Bubbleformer-S model;
#   - run_rollout: a rollout with the backbone and the head;
#   - rollout_statistics, distances_to_reference, compute_reward: how a rollout is scored (R2, R3 or R4);
#   - reference_statistics: the same statistics for a simulation, and the noise scale of each distance.
#
# For speed, all scores use the 128 x 128 grid:
#   - vapor mask: a coarse cell is vapor if more than half of its 4 x 4 fine cells are vapor (dfun > 0);
#   - departures: the frozen detector, with its pixel thresholds scaled to the coarse grid (detect_departures.py, cell=4);
#   - heat flux: the same wall heat-flux proxy as dwell_times_v2.py and compare_rollout_timing_v2.py.
# The final evaluation does not use these coarse scores. It uses the full 512 x 512 detector (compare_rollout_timing_v2.py).
import json
import numpy as np, torch
from detect_departures import find_departures
from bubbleformer_io import FIELDS, load_model
from flow_matching_head import FRAMES_PER_BUNDLE, average_to_128, sample_residual

DATA = "/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D"
FRAME_DT = 0.1           # time units between frames in BubbleML 2.0
FIRST_PREDICTED = 105    # Bubbleformer predicts from frame 105 when it starts from frames 100-104
SMOOTH_FRAMES = 10       # the heat flux is smoothed over this many frames before it is labeled quiet or active


def load_metadata(wall_temp, data=DATA):
    """The JSON metadata of one simulation file (heater size, wall temperature, fluid properties)."""
    return json.load(open(f"{data}/Twall_{wall_temp}.json"))


def heater_columns(metadata):
    """Boolean mask of the 128 columns of the coarse grid that lie over the heater."""
    heater = metadata["heater"]
    x = metadata["x_min"] + (np.arange(128) + 0.5) * (metadata["x_max"] - metadata["x_min"]) / 128
    return torch.from_numpy((x >= heater["xMin"]) & (x <= heater["xMax"]))


def wall_heat_flux(dfun_wall, temperature_wall, wall_temp):
    """Wall heat-flux proxy per frame, from the bottom row of the 128 x 128 fields over the heater.

    Liquid cells (dfun < 0) at the wall conduct heat in proportion to (wall temperature - fluid temperature).
    The formula is the same as in the other scripts. Inputs: (..., heater columns). Output: (...).
    """
    return (0.054 * (dfun_wall < 0) * (wall_temp - temperature_wall) / (0.0007 / 32)).mean(-1)


class Backbone:
    """The frozen Bubbleformer-S model. It runs in chunks so that a group of rollouts fits in GPU memory."""

    def __init__(self, checkpoint, device, chunk=4):
        self.model, shift, scale = load_model(checkpoint, device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.shift = torch.tensor([shift[k] for k in FIELDS], device=device).view(1, 1, 4, 1, 1)
        self.scale = torch.tensor([scale[k] for k in FIELDS], device=device).view(1, 1, 4, 1, 1)
        self.chunk = chunk

    @torch.no_grad()
    def __call__(self, frames, fluid):
        """frames: (B, 5, 4, 512, 512) in physical units; fluid: (B, 9). Returns the next 5 frames in physical units."""
        outputs = []
        for i in range(0, len(frames), self.chunk):
            normalized = (frames[i:i + self.chunk] - self.shift) / self.scale
            outputs.append(self.model(normalized, fluid[i:i + self.chunk]) * self.scale + self.shift)
        return torch.cat(outputs)


@torch.no_grad()
def run_rollout(backbone, head, normalizer, frames, fluid, wall_temp, metadata, n_bundles,
                bundles_to_keep=0, save_full_vapor=False, flow_steps=8):
    """Roll the model forward n_bundles bundles (5 frames each) from the start frames.

    Each step: the backbone predicts the next 5 frames; if a head is given, the head samples a residual on the
    coarse grid, and the residual (enlarged to 512 x 512) is added to the prediction. The result is the next input.

    frames: (B, 5, 4, 512, 512) start frames in physical units, on the GPU; fluid: (B, 9); wall_temp: (B,).
    bundles_to_keep: number of steps (the same steps for every rollout in the group) whose head noise, head sample
        and condition are kept for the post-training update.
    save_full_vapor: also return the 512 x 512 vapor mask and the 128 x 128 fields (for the evaluation files).
    Returns a dict:
        heat_flux (B, frames), vapor_128 (B, frames, 128, 128) bool, kept_steps (list of (noise, sample, condition)),
        and, if asked, vapor_512 (B, frames, 512, 512) bool and fields_128 (B, frames, 4, 128, 128).
    """
    columns = heater_columns(metadata).to(frames.device)
    keep_steps = set()
    if bundles_to_keep:
        keep_steps = set(np.random.choice(n_bundles, size=min(bundles_to_keep, n_bundles), replace=False).tolist())
    heat_flux, vapor_128, vapor_512, fields_128, kept_steps = [], [], [], [], []
    for step in range(n_bundles):
        prediction = backbone(frames, fluid)                                       # (B, 5, 4, 512, 512)
        if head is not None:
            condition = normalizer.build_condition(average_to_128(prediction), average_to_128(frames[:, -1]), wall_temp)
            noise, sample = sample_residual(head, condition, flow_steps)
            residual = normalizer.to_physical_residual(sample)                      # (B, 5, 4, 128, 128)
            residual_512 = torch.nn.functional.interpolate(
                residual.flatten(0, 1), scale_factor=4, mode="bilinear", align_corners=False)
            prediction = prediction + residual_512.view_as(prediction)
            if step in keep_steps:
                kept_steps.append((noise.half().cpu(), sample.half().cpu(), condition.half().cpu()))
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"rollout is not finite at step {step}")
        frames = prediction
        vapor = frames[:, :, 0] > 0                                                 # (B, 5, 512, 512)
        coarse = average_to_128(frames)
        heat_flux.append(wall_heat_flux(coarse[:, :, 0, 0, columns], coarse[:, :, 1, 0, columns],
                                        metadata["heater"]["wallTemp"]).cpu())
        vapor_128.append((average_to_128(vapor.float()) > 0.5).cpu())
        if save_full_vapor:
            vapor_512.append(vapor.cpu()); fields_128.append(coarse.half().cpu())
    result = {"heat_flux": torch.cat(heat_flux, 1).numpy(), "vapor_128": torch.cat(vapor_128, 1).numpy(),
              "kept_steps": kept_steps}
    if save_full_vapor:
        result["vapor_512"] = torch.cat(vapor_512, 1).numpy()
        result["fields_128"] = torch.cat(fields_128, 1).numpy()
    return result


def quiet_period_lengths(heat_flux, threshold):
    """Lengths, in frames, of the quiet periods: runs where the smoothed heat flux is below threshold.

    Runs cut by the start or the end of the signal are dropped, because their true length is unknown.
    """
    smooth = np.convolve(heat_flux, np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES, mode="valid")
    active = (smooth > threshold).astype(int)
    runs = np.split(active, np.flatnonzero(np.diff(active)) + 1)[1:-1]
    return np.array([len(run) for run in runs if run[0] == 0])


def rollout_statistics(heat_flux, vapor_128, quiet_threshold):
    """The statistics of one rollout (or one simulation stretch) that the rewards use, on the 128 x 128 grid."""
    events, _ = find_departures(vapor_128, FRAME_DT, cell=4)
    departure_frames = np.array(sorted({frame for frame, _ in events}))
    return {"departure_rate": 100 * len(departure_frames) / len(heat_flux),   # departures per 100 frames
            "waiting_times": np.diff(departure_frames),                        # frames between departures
            "quiet_periods": quiet_period_lengths(heat_flux, quiet_threshold),
            "heat_flux": heat_flux,
            "vapor_fraction": float(vapor_128.mean())}


def wasserstein_distance(a, b, n_quantiles=100):
    """1-Wasserstein distance between two samples, from n_quantiles quantiles. None if a sample has < 2 values."""
    if len(a) < 2 or len(b) < 2:
        return None
    levels = (np.arange(n_quantiles) + 0.5) / n_quantiles
    return float(np.mean(np.abs(np.quantile(a, levels) - np.quantile(b, levels))))


def distances_to_reference(statistics, reference):
    """Distance of each statistic from the reference simulation. 0 = the same as the simulation."""
    waiting = wasserstein_distance(statistics["waiting_times"], reference["waiting_times"])
    quiet = wasserstein_distance(statistics["quiet_periods"], reference["quiet_periods"])
    longest_possible = float(len(statistics["heat_flux"]))
    return {
        # snapshot statistics (R2)
        "heat_flux_histogram": wasserstein_distance(statistics["heat_flux"], reference["heat_flux"]),
        "vapor_fraction": abs(statistics["vapor_fraction"] - reference["vapor_fraction"]),
        # timing statistics (R3)
        "departure_rate": abs(statistics["departure_rate"] - reference["departure_rate"]),
        # With fewer than 2 departures or quiet periods, the distance is set to the rollout length (a large value).
        "waiting_times": waiting if waiting is not None else longest_possible,
        "quiet_periods": quiet if quiet is not None else longest_possible,
    }


REWARD_TERMS = {
    "R2": ("heat_flux_histogram", "vapor_fraction"),                         # snapshot statistics
    "R3": ("departure_rate", "waiting_times", "quiet_periods"),             # timing statistics
    "R4": ("heat_flux_histogram", "vapor_fraction", "departure_rate", "waiting_times", "quiet_periods"),
}


def compute_reward(reward_name, distances, noise_scales):
    """Reward = minus the sum of the reward's distances, each divided by its noise scale. Higher is better."""
    return -sum(distances[term] / noise_scales[term] for term in REWARD_TERMS[reward_name])


def reference_statistics(wall_temp, n_frames, data=DATA, n_samples=20, seed=0):
    """Statistics of one simulation, and the noise scale of each distance for a rollout of n_frames frames.

    Noise scale: the median distance between a random n_frames stretch of the simulation and the whole simulation.
    This is the distance that a perfect emulator would get by chance. It has a floor of 5% of the typical size of
    the quantity, so that no term gets a near-zero scale and dominates the reward.
    """
    metadata = load_metadata(wall_temp, data)
    columns = heater_columns(metadata).numpy()
    fields = np.load(f"{data}/Twall_{wall_temp}_fields128.npy", mmap_mode="r")[FIRST_PREDICTED:]
    vapor = np.load(f"{data}/Twall_{wall_temp}_vapor.npy", mmap_mode="r")
    heat_flux = wall_heat_flux(fields[:, 0, 0][:, columns].astype(np.float32),
                               fields[:, 1, 0][:, columns].astype(np.float32), metadata["heater"]["wallTemp"])
    vapor_128 = np.stack([vapor[t].reshape(128, 4, 128, 4).mean(axis=(1, 3)) > 0.5
                          for t in range(FIRST_PREDICTED, len(vapor))])
    quiet_threshold = float(np.median(np.convolve(heat_flux, np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES, mode="valid")))
    reference = rollout_statistics(heat_flux, vapor_128, quiet_threshold)

    rng = np.random.default_rng(seed)
    samples = {term: [] for term in REWARD_TERMS["R4"]}
    for _ in range(n_samples):
        start = rng.integers(0, len(heat_flux) - n_frames)
        stretch = rollout_statistics(heat_flux[start:start + n_frames], vapor_128[start:start + n_frames], quiet_threshold)
        for term, value in distances_to_reference(stretch, reference).items():
            samples[term].append(value)
    typical = {"heat_flux_histogram": float(np.std(reference["heat_flux"])),
               "vapor_fraction": reference["vapor_fraction"],
               "departure_rate": reference["departure_rate"],
               "waiting_times": float(np.mean(reference["waiting_times"])) if len(reference["waiting_times"]) else 1.0,
               "quiet_periods": float(np.mean(reference["quiet_periods"])) if len(reference["quiet_periods"]) else 1.0}
    noise_scales = {term: max(float(np.median(values)), 0.05 * typical[term], 1e-6) for term, values in samples.items()}
    return {"statistics": reference, "noise_scales": noise_scales, "quiet_threshold": quiet_threshold,
            "metadata": metadata}
