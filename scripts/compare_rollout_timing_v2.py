# Does Bubbleformer's own long rollout have the right bubble-departure timing?
# If the published model already matches the simulation's timing statistics, a timing reward (R3) has nothing to fix.
# If it does not, the size of the error is the motivation for R3.
#
# Usage (from the repo root, on a GPU node):
#   git clone https://github.com/HPCForge/Bubbleformer /pub/ynagaraj/Bubbleformer   (its pip install fails; use PYTHONPATH)
#   PYTHONPATH=/pub/ynagaraj/Bubbleformer python scripts/compare_rollout_timing_v2.py \
#       --ckpt <path>/Bubbleformer-S-PB-Subcooled.ckpt --twall 97 --neighbors 96 98
# The checkpoint is on Hugging Face: hpcforge/Bubbleformer, file Bubbleformer-S-PB-Subcooled.ckpt (232 MB).
#
# Steps:
#   1. Start from simulation frames 100-104 (Bubbleformer's start time) and roll the model forward 5 frames at a time.
#      Each prediction becomes the next input, as in Bubbleformer's scripts/inference.py.
#   2. Save the predicted vapor mask (full 512 x 512) and the 4 fields averaged to 128 x 128, like download_bubbleml_v2.py.
#   3. Compute the same statistics for the rollout and for the simulation over the same frames:
#      departures (with the frozen detector), the wall heat-flux signal, its autocorrelation,
#      and the mean heat flux around each departure.
#   4. Compare. For scale, also compare the two halves of the simulation (noise floor)
#      and the simulations at neighbouring wall temperatures (what a 1-2 degree error looks like).
#
# Needs in DATA: Twall_<T>.json, Twall_<T>_vapor.npy, Twall_<T>_fields128.npy (from download_bubbleml_v2.py),
# for the test file and each neighbour. The initial 5 frames are read from Hugging Face if Twall_<T>.hdf5 is not in DATA.
import argparse, json, os, time
import numpy as np, torch, h5py, fsspec
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from detect_departures import find_departures

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--twall", type=int, default=97)
p.add_argument("--frames", type=int, default=1895, help="frames to predict; 1895 reaches the last frame (2000)")
p.add_argument("--neighbors", type=int, nargs="*", default=[], help="wall temperatures to compare with, for scale")
p.add_argument("--data", default="/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D")
p.add_argument("--out", default="results/v2")
p.add_argument("--tag", default="", help="suffix for output and rollout file names, for example _L for Bubbleformer-L")
p.add_argument("--force", action="store_true", help="run the rollout again even if its files exist")
args = p.parse_args()

HF = "https://huggingface.co/datasets/hpcforge/BubbleML_2/resolve/main/PoolBoiling-Subcooled-FC72-2D"
FIELDS = ["dfun", "temperature", "velx", "vely"]   # Bubbleformer's fields, in its order
W, START = 5, 100                                  # Bubbleformer's time window and start frame
POOL = 4                                           # 512 -> 128, as in download_bubbleml_v2.py
MAX_LAG = 100                                      # frames (10 time units) for the autocorrelation
BEFORE, AFTER = 20, 50                             # frames before and after a departure for the event response
name = f"Twall_{args.twall}"
meta = json.load(open(f"{args.data}/{name}.json"))
first = START + W                                  # first predicted frame
frames = range(first, first + args.frames)


# ---------- 1-2. Rollout ----------

def load_model(path, device):
    """Build the model from a Bubbleformer checkpoint, as in Bubbleformer's scripts/inference.py."""
    from bubbleformer.models import get_model
    import importlib, pickle, types, torch._weights_only_unpickler as wu
    # Load safely: the file may create only allow-listed classes, so it cannot run other code while it is read.
    # torch's weights_only=True loader cannot be used, because it cannot fill the defaultdict in this checkpoint.
    # The allow list is torch's own safe list plus the classes this checkpoint needs for its omegaconf settings
    # (listed by torch.serialization.get_unsafe_globals_in_checkpoint).
    extra = ["omegaconf.listconfig.ListConfig", "omegaconf.dictconfig.DictConfig", "omegaconf.base.ContainerMetadata",
             "omegaconf.base.Metadata", "omegaconf.nodes.AnyNode", "collections.defaultdict", "typing.Any",
             "builtins.list", "builtins.dict", "builtins.int"]
    allowed = dict(wu._get_allowed_globals())
    for key in extra:
        mod, attr = key.rsplit(".", 1)
        allowed[key] = getattr(importlib.import_module(mod), attr)

    import _compat_pickle

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            # Old pickle protocols use Python 2 names (for example __builtin__.dict). Translate them first,
            # as the standard unpickler does.
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[(module, name)]
            elif module in _compat_pickle.IMPORT_MAPPING:
                module = _compat_pickle.IMPORT_MAPPING[module]
            key = f"{module}.{name}"
            if key not in allowed:
                raise pickle.UnpicklingError(f"checkpoint uses a class that is not allowed: {key}")
            return allowed[key]

    restricted = types.SimpleNamespace(Unpickler=Restricted, load=pickle.load, __name__="restricted_pickle")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, pickle_module=restricted)
    hyper_params = checkpoint["hyper_parameters"]
    params = dict(hyper_params["model_cfg"]["params"])
    params["input_fields"] = len(hyper_params["data_cfg"]["input_fields"])
    params["output_fields"] = len(hyper_params["data_cfg"]["output_fields"])
    params["time_window"] = hyper_params["data_cfg"]["time_window"]
    model = get_model(hyper_params["model_cfg"]["name"], **params)
    model.load_state_dict({k[len("model."):]: v for k, v in checkpoint["state_dict"].items()})  # drop the Lightning prefix
    field_shift, field_scale = hyper_params["normalization_constants"]
    return model.to(device).eval(), field_shift, field_scale


def fluid_params(metadata):
    """The 9 conditioning values, in the order of Bubbleformer's BubbleForecast dataset."""
    h = metadata["heater"]
    return torch.tensor([[metadata["inv_reynolds"], metadata["cpgas"], metadata["mugas"], metadata["rhogas"], metadata["thcogas"],
                          metadata["stefan"], metadata["prandtl"], h["nucWaitTime"], h["wallTemp"]]], dtype=torch.float32)


def initial_frames():
    """Frames START ... START+W-1 of the 4 fields at 512 x 512, shape (W, 4, 512, 512)."""
    local = f"{args.data}/{name}.hdf5"
    remote_file = None if os.path.exists(local) else fsspec.open(f"{HF}/{name}.hdf5", "rb", block_size=2**24).open()
    with h5py.File(local if remote_file is None else remote_file) as f:
        x = np.stack([f[k][START:START + W] for k in FIELDS], axis=1)
    if remote_file is not None:
        remote_file.close()
    return torch.from_numpy(x.astype(np.float32))


def average_cells(frames):
    """Average each 4 x 4 block of cells: (..., 512, 512) -> (..., 128, 128)."""
    *lead, height, width = frames.shape
    return frames.reshape(*lead, height // POOL, POOL, width // POOL, POOL).mean(axis=(-3, -1))


def rollout(vapor_path, fields_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, field_shift, field_scale = load_model(args.ckpt, device)
    shift = torch.tensor([field_shift[k] for k in FIELDS]).view(1, 4, 1, 1)
    scale = torch.tensor([field_scale[k] for k in FIELDS]).view(1, 4, 1, 1)
    x = ((initial_frames() - shift) / scale)[None].to(device)          # (1, W, 4, H, W), normalized
    conditioning = fluid_params(meta).to(device)
    n_bundles = -(-args.frames // W)                                 # round up
    vapor = np.zeros((n_bundles * W, 512, 512), dtype=bool)
    fields = np.zeros((n_bundles * W, 4, 128, 128), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for b in range(n_bundles):
            x = model(x, conditioning)                                       # the prediction is the next input
            y = (x[0].cpu() * scale + shift).numpy()                 # (W, 4, H, W), physical units
            if not np.isfinite(y).all():
                raise RuntimeError(f"rollout is not finite at bundle {b} (frame {first + b * W})")
            vapor[b * W:(b + 1) * W] = y[:, 0] > 0
            fields[b * W:(b + 1) * W] = average_cells(y)
            if b % 20 == 0:
                print(f"bundle {b} of {n_bundles}, {time.time() - t0:.0f} s", flush=True)
    np.save(vapor_path, vapor[:args.frames])
    np.save(fields_path, fields[:args.frames])


# ---------- 3. Statistics ----------

def heat_flux(fields, metadata):
    """Wall heat-flux signal, one value per frame.

    Same formula as Bubbleformer's utils/heatflux.py: liquid cells at the wall, over the heater,
    times (wall temperature - fluid temperature). Here it uses the 128 x 128 fields, so the bottom
    row is the mean of the 4 bottom rows at 512 x 512. The signal is a proxy: it is fair for comparing
    a rollout with the simulation, because both use the same formula, but it is not the exact heat flux.
    """
    h = metadata["heater"]
    x = metadata["x_min"] + (np.arange(128) + 0.5) * (metadata["x_max"] - metadata["x_min"]) / 128
    over_heater = (x >= h["xMin"]) & (x <= h["xMax"])
    dfun, temp = fields[:, 0, 0, over_heater].astype(np.float32), fields[:, 1, 0, over_heater].astype(np.float32)
    dx, lc = 1 / 32, 0.0007                        # Bubbleformer's constants, so the values have its scale
    return (0.054 * (dfun < 0) * (h["wallTemp"] - temp) / (dx * lc)).mean(axis=1)


def autocorrelation(flux):
    """Autocorrelation for lags 0 ... MAX_LAG, or up to half the signal length if the signal is short."""
    flux = flux - flux.mean()
    lags = range(min(MAX_LAG, len(flux) // 2) + 1)
    return np.array([np.dot(flux[:len(flux) - k], flux[k:]) for k in lags]) / np.dot(flux, flux)


def timing_statistics(vapor, fields, metadata, frame_dt):
    """All statistics for one stretch of frames. Departure frames are relative to the stretch."""
    events, _ = find_departures(vapor, frame_dt)
    departures = np.array(sorted({t for t, _ in events}))
    flux = heat_flux(fields, metadata)
    usable = departures[(departures >= BEFORE) & (departures + AFTER < len(flux))]
    response = np.mean([flux[d - BEFORE:d + AFTER] for d in usable], axis=0) if len(usable) else np.full(BEFORE + AFTER, np.nan)
    intervals = np.diff(departures)
    return {"departures": departures, "intervals": intervals, "flux": flux, "autocorrelation": autocorrelation(flux), "response": response,
            "vapor_fraction": float(vapor.mean())}


def wasserstein_1(a, b, n=200):
    """Wasserstein-1 distance between two 1D samples, from 200 quantiles: the average shift needed to match them."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    levels = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, levels) - np.quantile(b, levels))))


def results_column(signal, reference):
    """One column of the results table: the statistics of this signal, and its distances to the reference."""
    intervals, n_frames = signal["intervals"], len(signal["flux"])
    n_lag = min(len(signal["autocorrelation"]), len(reference["autocorrelation"]))
    return {
        "departures": len(signal["departures"]),
        "departures per 100 frames": 100 * len(signal["departures"]) / n_frames,
        "mean interval (frames)": intervals.mean() if len(intervals) else np.nan,
        "interval coefficient of variation": intervals.std() / intervals.mean() if len(intervals) else np.nan,
        "interval distance to simulation (frames)": wasserstein_1(intervals, reference["intervals"]),
        "heat-flux mean": signal["flux"].mean(),
        "heat-flux std": signal["flux"].std(),
        "heat-flux histogram distance to simulation": wasserstein_1(signal["flux"], reference["flux"]),
        "heat-flux autocorrelation RMS difference to simulation": float(np.sqrt(np.mean((signal["autocorrelation"][:n_lag] - reference["autocorrelation"][:n_lag]) ** 2))),
        "heat flux around departures, RMS difference to simulation": float(np.sqrt(np.mean((signal["response"] - reference["response"]) ** 2))),
        "vapor fraction": signal["vapor_fraction"],
    }


def simulation_statistics(twall, frame_slice):
    """Simulation statistics for wall temperature twall over the frame slice frame_slice."""
    metadata = json.load(open(f"{args.data}/Twall_{twall}.json"))
    vapor = np.load(f"{args.data}/Twall_{twall}_vapor.npy", mmap_mode="r")[frame_slice]
    fields = np.load(f"{args.data}/Twall_{twall}_fields128.npy", mmap_mode="r")[frame_slice]
    return timing_statistics(np.asarray(vapor), np.asarray(fields), metadata, frame_dt)


# ---------- Main ----------

T_total = np.load(f"{args.data}/{name}_vapor.npy", mmap_mode="r").shape[0]
frame_dt = (meta["t_final"] - meta["t_initial"]) / (T_total - 1)
vapor_path, fields_path = f"{args.data}/{name}_rollout{args.tag}_vapor.npy", f"{args.data}/{name}_rollout{args.tag}_fields128.npy"
if args.force or not os.path.exists(vapor_path):
    rollout(vapor_path, fields_path)
roll_vapor, roll_fields = np.load(vapor_path), np.load(fields_path)
assert len(roll_vapor) == args.frames, "saved rollout has a different length; run again with --force"

frame_slice = slice(frames.start, frames.stop)
half = frames.start + args.frames // 2
sim = simulation_statistics(args.twall, frame_slice)
roll = timing_statistics(roll_vapor, roll_fields, meta, frame_dt)
columns = {"Bubbleformer rollout": results_column(roll, sim), f"simulation Twall {args.twall}": results_column(sim, sim)}
# Noise floor: the second half of the simulation against the first half.
sim_a, sim_b = simulation_statistics(args.twall, slice(frames.start, half)), simulation_statistics(args.twall, slice(half, frames.stop))
columns["simulation, second half vs first half"] = results_column(sim_b, sim_a)
for t in args.neighbors:
    columns[f"simulation Twall {t}"] = results_column(simulation_statistics(t, frame_slice), sim)

# Table: one row per statistic, one column per signal.
keys = list(next(iter(columns.values())))
lines = [f"# Bubbleformer rollout vs simulation, Twall {args.twall}", "",
         f"Frames {frames.start}-{frames.stop - 1} ({args.frames} frames, {args.frames * frame_dt:.1f} time units). "
         f"Rollout starts from simulation frames {START}-{START + W - 1}. Checkpoint: `{os.path.basename(args.ckpt)}`.", "",
         "| Statistic | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
for k in keys:
    lines.append(f"| {k} | " + " | ".join(f"{c[k]:.3g}" for c in columns.values()) + " |")
table = "\n".join(lines)
print(table)
os.makedirs(args.out, exist_ok=True)
out = f"{args.out}/rollout_timing_{name}{args.tag}"
open(f"{out}.md", "w").write(table + "\n")
json.dump({c: {k: float(v) for k, v in vals.items()} for c, vals in columns.items()}, open(f"{out}.json", "w"), indent=1)

# Figure: heat flux, departure raster, interval histogram, autocorrelation, event response.
lag = np.arange(len(sim["autocorrelation"])) * frame_dt
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
tt = np.arange(frames.start, frames.stop)
ax[0].plot(tt, sim["flux"], lw=0.6, label="simulation")
ax[0].plot(tt, roll["flux"], lw=0.6, label="rollout")
top = max(sim["flux"].max(), roll["flux"].max())
ax[0].vlines(frames.start + sim["departures"], top, top * 1.05, lw=0.5, color="C0")
ax[0].vlines(frames.start + roll["departures"], top * 1.05, top * 1.1, lw=0.5, color="C1")
ax[0].set(title="wall heat flux (ticks: departures)", xlabel="frame"); ax[0].legend(fontsize=8)
bins = np.arange(0, max(sim["intervals"].max(initial=1), roll["intervals"].max(initial=1)) + 2, 2)
ax[1].hist(sim["intervals"], bins, histtype="step", density=True, label="simulation")
ax[1].hist(roll["intervals"], bins, histtype="step", density=True, label="rollout")
ax[1].set(title="time between departures", xlabel="frames")
ax[2].plot(lag, sim["autocorrelation"], label="simulation"); ax[2].plot(lag, roll["autocorrelation"], label="rollout")
ax[2].set(title="heat-flux autocorrelation", xlabel="lag (time units)")
rel = np.arange(-BEFORE, AFTER)
ax[3].plot(rel, sim["response"], label="simulation"); ax[3].plot(rel, roll["response"], label="rollout")
ax[3].axvline(0, color="k", lw=0.5)
ax[3].set(title="mean heat flux around a departure", xlabel="frames from departure")
plt.tight_layout(); plt.savefig(f"{out}.png", dpi=110)
