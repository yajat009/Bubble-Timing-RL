# T3 observability probe on BubbleML 2.0 (see analysis/TEST_PLAN.md).
# Question: do Bubbleformer's 5 input frames contain the time to the next bubble departure?
# Run from the repo root: python scripts/probe_v2.py
#
# Data: the 4 Bubbleformer fields (dfun, temperature, velx, vely) at 128 x 128, from scripts/download_v2.py.
# Split: Bubbleformer's own. Train on the 16 FC-72 training files, test on its validation file (Twall 97).
# Target: frames from the last input frame until the next departure (any nucleation site).
# Models (letters as in the test plan):
#   (a) mean only: always predicts the training mean
#   (b) time only: the true number of frames since the last departure
#   (c) frames: 5 frames x 4 fields
#   (d) frames + time
# Every setting below, including the pass criteria, is fixed before the first run.
import json, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D"
TRAIN = [86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116]
TEST = [97]
W = 5            # frames per input window
START = 100      # Bubbleformer skips the first 100 frames
SEEDS = range(5)
EPOCHS, LR, BS = 10, 1e-3, 64
TRAIN_STRIDE = 5 # use every 5th training window; neighbours share 4 of 5 frames. Test windows use every window.
SCALE = 10.0     # targets and the time input are divided by this, so the network works with numbers near 1
dev = "cuda"

def load(t):
    frames = torch.from_numpy(np.load(f"{D}/Twall_{t}_fields128.npy"))       # (T, 4, 128, 128) float16
    deps = np.load(f"results/v2/departures_Twall_{t}.npy")
    idx, since, until, gap = [], [], [], []
    for i in range(START + W - 1, len(frames)):                            # window = frames i-W+1 ... i
        past, future = deps[deps <= i], deps[deps > i]
        if len(past) == 0 or len(future) == 0:                             # need both, so all models share samples
            continue
        idx.append(i); since.append(i - past.max()); until.append(future.min() - i)
        gap.append(future.min() - past.max())                              # length of the interval this window is in
    return frames, np.array(idx), np.array(since, np.float32), np.array(until, np.float32), np.array(gap, np.float32)

data = {t: load(t) for t in TRAIN + TEST}
print({t: len(d[1]) for t, d in data.items()}, "windows per file", flush=True)

# Per-field mean and std from the training files only.
sample = torch.cat([data[t][0][START::20].float() for t in TRAIN])
MEAN = sample.mean(dim=(0, 2, 3)).view(1, 1, 4, 1, 1).to(dev)
STD = sample.std(dim=(0, 2, 3)).view(1, 1, 4, 1, 1).to(dev)
# All files in one GPU tensor (about 260 MB per file in float16). OFFSET[t] is where file t starts.
OFFSET, n = {}, 0
for t in data:
    OFFSET[t] = n; n += len(data[t][0])
ALL = torch.cat([data[t][0] for t in data]).to(dev)
STEPS = torch.arange(-W + 1, 1, device=dev)

def split(files, stride=1):
    # Returns, for every window: its global frame index in ALL, time since, time until, and interval length.
    g = np.concatenate([OFFSET[t] + data[t][1][::stride] for t in files])
    cat = lambda k: np.concatenate([data[t][k][::stride] for t in files])
    return g, cat(2), cat(3), cat(4)

def windows(g):
    # Gather frames g-4 ... g for each window and stack them as 20 channels (5 frames x 4 fields).
    x = ALL[torch.as_tensor(g, device=dev)[:, None] + STEPS].float()
    return ((x - MEAN) / STD).flatten(1, 2)

class Net(nn.Module):
    def __init__(self, use_frames, use_since):
        super().__init__()
        self.use_frames, self.use_since = use_frames, use_since
        self.conv = nn.Sequential(
            nn.Conv2d(4 * W, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 * use_frames + use_since, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, since):
        parts = []
        if self.use_frames: parts.append(self.conv(x))
        if self.use_since: parts.append(since[:, None] / SCALE)
        return self.head(torch.cat(parts, 1))[:, 0] * SCALE

def predict(m, g, s):
    out = []
    with torch.no_grad():
        for b in range(0, len(g), 256):
            x = windows(g[b:b + 256]) if m.use_frames else None
            out.append(m(x, torch.from_numpy(s[b:b + 256]).to(dev)).cpu())
    return torch.cat(out).numpy()

def train_eval(tr, te, use_frames, use_since, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(use_frames, use_since).to(dev)
    opt = torch.optim.Adam(m.parameters(), LR)
    g, s, y, _ = tr
    for _ in range(EPOCHS):
        for b in np.array_split(np.random.permutation(len(g)), len(g) // BS):
            x = windows(g[b]) if use_frames else None
            pred = m(x, torch.from_numpy(s[b]).to(dev))
            loss = (pred - torch.from_numpy(y[b]).to(dev)).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return predict(m, te[0], te[1])                                 # no early stopping on test data

tr, te = split(TRAIN, TRAIN_STRIDE), split(TEST)
y_te, gap_te = te[2], te[3]
# Irregular windows: inside an interval more than 1 std from the mean training interval.
train_gaps = np.concatenate([np.diff(np.load(f"results/v2/departures_Twall_{t}.npy")) for t in TRAIN])
irregular = np.abs(gap_te - train_gaps.mean()) > train_gaps.std(ddof=1)
print(f"train windows {len(tr[0])}, test windows {len(te[0])}, irregular test windows {irregular.sum()}; "
      f"train interval mean {train_gaps.mean():.1f} std {train_gaps.std(ddof=1):.1f} frames", flush=True)

MODELS = {"(b) time only": (0, 1), "(c) frames": (1, 0), "(d) frames + time": (1, 1)}
mae = {"(a) mean only": np.full(len(SEEDS), np.abs(y_te - tr[2].mean()).mean())}
mae_irr = {"(a) mean only": np.full(len(SEEDS), np.abs(y_te - tr[2].mean())[irregular].mean())}
for k, (uf, us) in MODELS.items():
    mae[k], mae_irr[k] = np.zeros(len(SEEDS)), np.zeros(len(SEEDS))
    for s in SEEDS:
        err = np.abs(train_eval(tr, te, uf, us, s) - y_te)
        mae[k][s], mae_irr[k][s] = err.mean(), err[irregular].mean()
        print(f"{k} seed {s}: MAE {mae[k][s]:.2f}, irregular {mae_irr[k][s]:.2f}", flush=True)

print(f"\nTest MAE in frames on Twall 97 (mean +- sample std over {len(SEEDS)} seeds):")
for k in mae:
    print(f"  {k:20s} all {mae[k].mean():.2f} +- {mae[k].std(ddof=1):.2f}   irregular {mae_irr[k].mean():.2f} +- {mae_irr[k].std(ddof=1):.2f}")

# Pass criteria from analysis/TEST_PLAN.md (T3). sd = sample std over seeds of model (c).
a, b, c, d = (mae[k].mean() for k in mae)
sd = mae["(c) frames"].std(ddof=1)
b_irr, c_irr = mae_irr["(b) time only"].mean(), mae_irr["(c) frames"].mean()
crit = {"1: (c) beats (a) by more than 2 sd": a - c > 2 * sd,
        "2: |(d) - (c)| within 1 sd": abs(d - c) <= sd,
        "3: on irregular intervals, (c) beats (b) by more than 2 sd": b_irr - c_irr > 2 * sd}
print(f"\nsd of (c) = {sd:.3f}")
for name, ok in crit.items():
    print(f"  criterion {name}: {'PASS' if ok else 'FAIL'}")

json.dump({"mae": {k: v.tolist() for k, v in mae.items()}, "mae_irregular": {k: v.tolist() for k, v in mae_irr.items()},
           "sd_c": float(sd), "criteria": {k: bool(v) for k, v in crit.items()},
           "test_windows": int(len(y_te)), "irregular_windows": int(irregular.sum())},
          open("results/v2/probe_v2.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, (res, title) in enumerate([(mae, "all test windows"), (mae_irr, "irregular intervals only")]):
    ax[j].bar(list(res), [v.mean() for v in res.values()], yerr=[v.std(ddof=1) for v in res.values()])
    ax[j].set(ylabel="test MAE (frames)", title=f"Twall 97, {title}"); ax[j].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig("results/v2/probe_v2.png", dpi=110)
