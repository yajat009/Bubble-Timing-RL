# Test C: does a longer input window help predict the next bubble departure?
# T3 found that 5 frames predict the time to the next departure no better than a constant guess.
# If more history helps, the data has memory beyond Bubbleformer's 5-frame window, and a memory head may matter.
# If no window helps, departures are effectively memoryless at every window we test.
#
# Usage (from the repo root, on a GPU node): python scripts/window_sweep_v2.py --windows 1 2 5 10 20
#
# Setup (same as the T3 probe, except for the window):
#   - Input: the last W frames x 4 fields at 128 x 128. Target: frames until the next departure.
#   - Train on Bubbleformer's 16 training files, test on Twall 97.
#   - Every window length uses the same samples: windows end at frame >= 100 + 20 - 1, so even W = 20 fits.
#   - Baseline: the median training target (the best constant for absolute error).
import argparse, json, os
import numpy as np, torch, torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--windows", type=int, nargs="+", default=[1, 2, 5, 10, 20])
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--data", default="/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D")
p.add_argument("--out", default="results/v2")
args = p.parse_args()

TRAIN = [86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116]
TEST = [97]
START, MAX_W = 100, max(args.windows)
TRAIN_STRIDE, LR, BS, SCALE = 5, 1e-3, 64, 10.0
dev = "cuda" if torch.cuda.is_available() else "cpu"


def departure_frames(t):
    """Departure frames of one file. The file name changed in a repo rename, so accept both names."""
    for f in (f"results/v2/departure_frames_Twall_{t}.npy", f"results/v2/departures_Twall_{t}.npy"):
        if os.path.exists(f):
            return np.load(f)
    raise FileNotFoundError(f"no departure file for Twall {t}")


def load(t):
    frames = torch.from_numpy(np.load(f"{args.data}/Twall_{t}_fields128.npy"))   # (T, 4, 128, 128) float16
    deps = departure_frames(t)
    last, until = [], []
    for i in range(START + MAX_W - 1, len(frames)):
        future = deps[deps > i]
        if len(deps[deps <= i]) and len(future):              # same rule as T3: a previous and a next departure
            last.append(i); until.append(future.min() - i)
    return frames, np.array(last, np.int64), np.array(until, np.float32)


data = {t: load(t) for t in TRAIN + TEST}
sample = torch.cat([data[t][0][START::20].float() for t in TRAIN])
MEAN = sample.mean(dim=(0, 2, 3)).view(1, 1, 4, 1, 1).to(dev)
STD = sample.std(dim=(0, 2, 3)).view(1, 1, 4, 1, 1).to(dev)
OFFSET, n = {}, 0
for t in data:
    OFFSET[t] = n; n += len(data[t][0])
ALL = torch.cat([data[t][0] for t in data]).to(dev)


def split(files, stride=1):
    g = np.concatenate([OFFSET[t] + data[t][1][::stride] for t in files])
    y = np.concatenate([data[t][2][::stride] for t in files])
    return g, y


def windows(g, w):
    """Frames g-w+1 ... g for each sample, stacked as 4*w channels."""
    steps = torch.arange(-w + 1, 1, device=dev)
    x = ALL[torch.as_tensor(g, device=dev)[:, None] + steps].float()
    return ((x - MEAN) / STD).flatten(1, 2)


def net(w):
    return nn.Sequential(
        nn.Conv2d(4 * w, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(dev)


(g_tr, y_tr), (g_te, y_te) = split(TRAIN, TRAIN_STRIDE), split(TEST)
median_mae = float(np.abs(y_te - np.median(y_tr)).mean())
print(f"train {len(g_tr)}, test {len(g_te)} windows; median baseline MAE {median_mae:.2f}", flush=True)

results = {}
for w in args.windows:
    maes = []
    for seed in range(args.seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        m = net(w)
        opt = torch.optim.Adam(m.parameters(), LR)
        for _ in range(args.epochs):
            for b in np.array_split(np.random.permutation(len(g_tr)), len(g_tr) // BS):
                pred = m(windows(g_tr[b], w))[:, 0] * SCALE
                loss = (pred - torch.from_numpy(y_tr[b]).to(dev)).abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pred = torch.cat([m(windows(g_te[b:b + 256], w))[:, 0].cpu() * SCALE for b in range(0, len(g_te), 256)])
        maes.append(float(np.abs(pred.numpy() - y_te).mean()))
        print(f"window {w} seed {seed}: MAE {maes[-1]:.2f}", flush=True)
    results[w] = maes

lines = ["# Window sweep: frames until the next departure, Twall 97", "",
         f"Median baseline MAE: {median_mae:.2f} frames. {args.seeds} seeds, {args.epochs} epochs. Lower is better.", "",
         "| Window (frames) | Window (time units) | MAE mean | MAE std | Gain over median |", "|---|---|---|---|---|"]
for w, maes in results.items():
    lines.append(f"| {w} | {w / 10:g} | {np.mean(maes):.2f} | {np.std(maes):.2f} | {median_mae - np.mean(maes):+.2f} |")
table = "\n".join(lines)
print(table)
os.makedirs(args.out, exist_ok=True)
open(f"{args.out}/window_sweep_Twall_97.md", "w").write(table + "\n")
json.dump({"median_mae": median_mae, "mae": {str(w): v for w, v in results.items()}},
          open(f"{args.out}/window_sweep_Twall_97.json", "w"), indent=1)
