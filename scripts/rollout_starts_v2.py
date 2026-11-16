# Test D: does Bubbleformer's rollout lose bubble activity from every start state, or only from frame 100?
# T6 used one start (frames 100-104). This script repeats a shorter rollout from several other start frames
# and compares departures and heat flux with the simulation over the same frames.
#
# Usage (from the repo root, on a GPU node):
#   PYTHONPATH=/pub/ynagaraj/Bubbleformer python scripts/rollout_starts_v2.py \
#       --ckpt /pub/ynagaraj/bubbleformer_ckpt/Bubbleformer-S-PB-Subcooled.ckpt --starts 250 500 750 995 --frames 1000
#
# For each start frame s:
#   1. Give the model simulation frames s ... s+4 and roll it forward to `--frames` predicted frames.
#   2. Save the rollout as <DATA>/Twall_<T>_rollout_start<s>_vapor.npy and _fields128.npy (test B reads these).
#   3. Find departures in the rollout and in the simulation over the same frames, with the frozen detector.
#   4. Report the departure rate, interval statistics, the interval W1 distance, and the mean heat flux.
# The 5 start frames are read from Hugging Face if Twall_<T>.hdf5 is not in DATA.
import argparse, json, os, time
import numpy as np, torch
from detect_departures import find_departures
from bubbleformer_io import FIELDS, load_model, fluid_params, read_frames

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--twall", type=int, default=97)
p.add_argument("--starts", type=int, nargs="+", default=[250, 500, 750, 995])
p.add_argument("--frames", type=int, default=1000, help="frames to predict from each start")
p.add_argument("--data", default="/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D")
p.add_argument("--out", default="results/v2")
args = p.parse_args()

W, POOL = 5, 4
name = f"Twall_{args.twall}"
meta = json.load(open(f"{args.data}/{name}.json"))
sim_vapor = np.load(f"{args.data}/{name}_vapor.npy", mmap_mode="r")
sim_fields = np.load(f"{args.data}/{name}_fields128.npy", mmap_mode="r")
frame_dt = (meta["t_final"] - meta["t_initial"]) / (len(sim_vapor) - 1)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model, diff, div = load_model(args.ckpt, dev)
shift = torch.tensor([diff[k] for k in FIELDS]).view(1, 4, 1, 1)
scale = torch.tensor([div[k] for k in FIELDS]).view(1, 4, 1, 1)
cond = fluid_params(meta).to(dev)


def shrink(a):
    *lead, h, w = a.shape
    return a.reshape(*lead, h // POOL, POOL, w // POOL, POOL).mean(axis=(-3, -1))


def heat_flux(fields):
    """Wall heat-flux proxy per frame (same formula as the rollout comparison script)."""
    h = meta["heater"]
    x = meta["x_min"] + (np.arange(128) + 0.5) * (meta["x_max"] - meta["x_min"]) / 128
    over = (x >= h["xMin"]) & (x <= h["xMax"])
    dfun, temp = fields[:, 0, 0, over].astype(np.float32), fields[:, 1, 0, over].astype(np.float32)
    return (0.054 * (dfun < 0) * (h["wallTemp"] - temp) / (0.0007 / 32)).mean(axis=1)


def w1(a, b, n=200):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def departure_stats(vapor, fields, ref_gaps=None):
    events, _ = find_departures(vapor, frame_dt)
    deps = np.array(sorted({t for t, _ in events}))
    gaps = np.diff(deps)
    row = {"departures": len(deps), "departures per 100 frames": 100 * len(deps) / len(vapor),
           "mean interval (frames)": gaps.mean() if len(gaps) else np.nan,
           "interval CV": gaps.std() / gaps.mean() if len(gaps) else np.nan,
           "heat flux mean": float(heat_flux(fields).mean()), "vapor fraction": float(vapor.mean())}
    if ref_gaps is not None:
        row["interval W1 to sim (frames)"] = w1(gaps, ref_gaps)
    return row, gaps


cols = {}
for s in args.starts:
    n = min(args.frames, len(sim_vapor) - (s + W))
    n_bundles = -(-n // W)
    vap_path = f"{args.data}/{name}_rollout_start{s}_vapor.npy"
    fld_path = f"{args.data}/{name}_rollout_start{s}_fields128.npy"
    if not os.path.exists(vap_path):
        x = ((read_frames(args.data, name, s, s + W) - shift) / scale)[None].to(dev)
        vapor = np.zeros((n_bundles * W, 512, 512), dtype=bool)
        fields = np.zeros((n_bundles * W, 4, 128, 128), dtype=np.float16)
        t0 = time.time()
        with torch.no_grad():
            for b in range(n_bundles):
                x = model(x, cond)                                       # the prediction is the next input
                y = (x[0].cpu() * scale + shift).numpy()
                if not np.isfinite(y).all():
                    raise RuntimeError(f"start {s}: rollout is not finite at bundle {b}")
                vapor[b * W:(b + 1) * W] = y[:, 0] > 0
                fields[b * W:(b + 1) * W] = shrink(y)
        print(f"start {s}: rolled out {n} frames in {time.time() - t0:.0f} s", flush=True)
        np.save(vap_path, vapor[:n]); np.save(fld_path, fields[:n])
    roll_vapor, roll_fields = np.load(vap_path), np.load(fld_path)
    sl = slice(s + W, s + W + n)
    sim_row, sim_gaps = departure_stats(np.asarray(sim_vapor[sl]), np.asarray(sim_fields[sl]))
    roll_row, _ = departure_stats(roll_vapor, roll_fields, sim_gaps)
    cols[f"start {s}: sim"], cols[f"start {s}: rollout"] = sim_row, roll_row
    print(f"start {s}: sim {sim_row['departures']} departures, rollout {roll_row['departures']}", flush=True)

keys = ["departures", "departures per 100 frames", "mean interval (frames)", "interval CV",
        "interval W1 to sim (frames)", "heat flux mean", "vapor fraction"]
lines = [f"# Bubbleformer rollouts from several start frames, Twall {args.twall}", "",
         f"Up to {args.frames} predicted frames per start. Checkpoint: `{os.path.basename(args.ckpt)}`.", "",
         "| Statistic | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
for k in keys:
    lines.append(f"| {k} | " + " | ".join(f"{c[k]:.3g}" if k in c else "-" for c in cols.values()) + " |")
table = "\n".join(lines)
print(table)
os.makedirs(args.out, exist_ok=True)
out = f"{args.out}/rollout_starts_{name}"
open(f"{out}.md", "w").write(table + "\n")
json.dump({c: {k: float(v) for k, v in row.items()} for c, row in cols.items()}, open(f"{out}.json", "w"), indent=1)
