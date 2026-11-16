# Test B: how long does boiling stay quiet or active, in the simulation and in Bubbleformer rollouts?
# The T6 rollout had long quiet stretches that the simulation does not have. This test measures that directly,
# without the departure detector, and checks whether the quiet/active switching is memoryless.
#
# Usage (from the repo root; CPU is enough):
#   python scripts/dwell_times_v2.py --twall 97 --rollouts Twall_97_rollout Twall_97_rollout_start600 ...
# Each rollout name refers to <DATA>/<name>_fields128.npy and <DATA>/<name>_meta.json (or the T6 default).
#
# Steps:
#   1. Wall heat-flux signal per frame (the same proxy formula as the rollout comparison script).
#   2. Smooth it over 1 time unit (10 frames). Label each frame "active" if the smoothed signal is above
#      the simulation's median, else "quiet". The threshold comes from the simulation only.
#   3. Dwell time = length of each run of quiet or active frames. Compare the dwell-time distributions (W1)
#      of each rollout with the simulation, and with a noise floor (simulation 2nd half vs 1st half).
#   4. Markov check (Chapman-Kolmogorov): for a lag of L frames, compare the 2L-frame transition matrix with the
#      L-frame matrix squared. A large difference means the quiet/active switching has memory at that lag.
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--twall", type=int, default=97)
p.add_argument("--rollouts", nargs="*", default=None,
               help="rollout names in DATA; default: the T6 rollout, Twall_<T>_rollout")
p.add_argument("--data", default="/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D")
p.add_argument("--out", default="results/v2")
args = p.parse_args()

FIRST = 105          # first predicted frame of the T6 rollout (start frames 100-104)
SMOOTH = 10          # frames (1 time unit)
LAGS = [5, 10, 20]   # frames, for the Markov check
name = f"Twall_{args.twall}"
meta = json.load(open(f"{args.data}/{name}.json"))


def heat_flux(fields, j):
    """Wall heat-flux proxy per frame, from 128 x 128 fields (same formula as the rollout comparison script)."""
    h = j["heater"]
    x = j["x_min"] + (np.arange(128) + 0.5) * (j["x_max"] - j["x_min"]) / 128
    over_heater = (x >= h["xMin"]) & (x <= h["xMax"])
    dfun = fields[:, 0, 0, over_heater].astype(np.float32)
    temp = fields[:, 1, 0, over_heater].astype(np.float32)
    return (0.054 * (dfun < 0) * (h["wallTemp"] - temp) / (0.0007 / 32)).mean(axis=1)


def states(q, threshold):
    """1 = active, 0 = quiet, after smoothing over SMOOTH frames."""
    smooth = np.convolve(q, np.ones(SMOOTH) / SMOOTH, mode="valid")
    return (smooth > threshold).astype(int)


def dwell_times(s):
    """Lengths of the runs of equal values. Returns (quiet runs, active runs). Runs cut by the ends are dropped."""
    edges = np.flatnonzero(np.diff(s)) + 1
    runs = np.split(s, edges)[1:-1]
    return (np.array([len(r) for r in runs if r[0] == 0]), np.array([len(r) for r in runs if r[0] == 1]))


def markov_gap(s, lag):
    """Max |P(2 lag) - P(lag)^2| over the 2 x 2 transition matrix. Near 0 = memoryless at this lag."""
    def matrix(k):
        m = np.zeros((2, 2))
        np.add.at(m, (s[:-k], s[k:]), 1)
        return m / m.sum(axis=1, keepdims=True).clip(min=1)
    one = matrix(lag)
    return float(np.abs(matrix(2 * lag) - one @ one).max())


def w1(a, b, n=200):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def summary(s, ref_quiet, ref_active):
    quiet, active = dwell_times(s)
    row = {"frames": len(s), "active share": s.mean(),
           "quiet runs": len(quiet), "mean quiet dwell (frames)": quiet.mean() if len(quiet) else np.nan,
           "longest quiet dwell (frames)": quiet.max() if len(quiet) else np.nan,
           "mean active dwell (frames)": active.mean() if len(active) else np.nan,
           "quiet dwell W1 to sim": w1(quiet, ref_quiet), "active dwell W1 to sim": w1(active, ref_active)}
    for lag in LAGS:
        row[f"Markov gap, lag {lag}"] = markov_gap(s, lag)
    return row, quiet


# Simulation over the same frames as the rollouts, and its threshold.
sim_fields = np.load(f"{args.data}/{name}_fields128.npy", mmap_mode="r")
sim_q = heat_flux(np.asarray(sim_fields[FIRST:]), meta)
threshold = np.median(np.convolve(sim_q, np.ones(SMOOTH) / SMOOTH, mode="valid"))
sim_s = states(sim_q, threshold)
sim_quiet, sim_active = dwell_times(sim_s)
half = len(sim_s) // 2

cols, quiets = {}, {}
cols["simulation"], quiets["simulation"] = summary(sim_s, sim_quiet, sim_active)
# Noise floor: 2nd half against the 1st half.
a_quiet, a_active = dwell_times(sim_s[:half])
cols["sim 2nd half vs 1st half"], _ = summary(sim_s[half:], a_quiet, a_active)
for r in args.rollouts or [f"{name}_rollout"]:
    q = heat_flux(np.load(f"{args.data}/{r}_fields128.npy"), meta)
    cols[r], quiets[r] = summary(states(q, threshold), sim_quiet, sim_active)

keys = list(cols["simulation"])
lines = [f"# Quiet and active dwell times, Twall {args.twall}", "",
         f"Heat flux smoothed over {SMOOTH} frames; active = above the simulation's median ({threshold:.4g}). "
         "W1 columns compare with the simulation (0 = same). Markov gap: max |P(2L) - P(L)^2|.", "",
         "| Statistic | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
for k in keys:
    lines.append(f"| {k} | " + " | ".join(f"{c[k]:.3g}" for c in cols.values()) + " |")
table = "\n".join(lines)
print(table)
os.makedirs(args.out, exist_ok=True)
out = f"{args.out}/dwell_times_{name}"
open(f"{out}.md", "w").write(table + "\n")
json.dump({c: {k: float(v) for k, v in row.items()} for c, row in cols.items()}, open(f"{out}.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(6, 4))
bins = np.arange(0, max(max(len(v) and v.max() for v in quiets.values()), 1) + 20, 10)
for label, v in quiets.items():
    ax.hist(v, bins, histtype="step", density=True, label=label)
ax.set(xlabel="quiet dwell time (frames)", ylabel="density", yscale="log", title=f"Quiet dwell times, Twall {args.twall}")
ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{out}.png", dpi=110)
