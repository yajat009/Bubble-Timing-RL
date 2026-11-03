# BubbleML v1 only. Run from the repo root.
# Feasibility probe: can 5 consecutive frames predict the time to the next bubble departure?
# Models, tested leave-one-file-out (each test file is a wall temperature the model never saw):
#   baseline: always predict the mean training target
#   (a) frames only
#   (b) frames + true time since the last departure
#   (c) time since the last departure only (no frames)
import sys, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Folder with the *_vapor.npy masks that detect_departures.py wrote.
D = sys.argv[1] if len(sys.argv) > 1 else "/pub/ynagaraj/bubbleml/PoolBoiling-SubCooled-FC72-2D"
FILES = ["Twall-100", "Twall-103", "Twall-106", "Twall-110"]
W = 5                          # frames per input window
SEEDS = range(5)
EPOCHS, LR, BS = 40, 1e-3, 32
dev = "cuda"

def load(name):
    v = np.load(f"{D}/{name}_vapor.npy").astype(np.float32)   # vapor mask, shape (T, 384, 384)
    T = len(v)
    v = v.reshape(T, 96, 4, 96, 4).mean(axis=(2, 4))          # shrink each frame 4x
    deps = np.load(f"results/v1/departures_{name}.npy")
    X, since, until = [], [], []
    for t in range(W - 1, T):                                 # window = frames t-W+1 ... t
        past, future = deps[deps <= t], deps[deps > t]
        # Keep only windows with a previous and a next departure,
        # so every model is scored on the same samples.
        if len(past) == 0 or len(future) == 0:
            continue
        X.append(v[t - W + 1:t + 1]); since.append(t - past.max()); until.append(future.min() - t)
    return [torch.tensor(np.array(a), dtype=torch.float32) for a in (X, since, until)]

class Net(nn.Module):
    def __init__(self, use_frames, use_since):
        super().__init__()
        self.use_frames, self.use_since = use_frames, use_since
        self.conv = nn.Sequential(
            nn.Conv2d(W, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 * use_frames + use_since, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, since):
        parts = []
        if self.use_frames: parts.append(self.conv(x))
        if self.use_since: parts.append(since[:, None] / 10)   # scale to a size similar to the CNN features
        return self.head(torch.cat(parts, 1))[:, 0] * 10      # the network predicts in units of 10 frames

def train_eval(tr, te, use_frames, use_since, seed):
    torch.manual_seed(seed)
    m = Net(use_frames, use_since).to(dev)
    opt = torch.optim.Adam(m.parameters(), LR)
    X, s, y = [a.to(dev) for a in tr]
    for _ in range(EPOCHS):
        for i in torch.randperm(len(X), device=dev).split(BS):
            loss = (m(X[i], s[i]) - y[i]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Xt, st, yt = [a.to(dev) for a in te]
        return (m(Xt, st) - yt).abs().mean().item()          # test MAE, no early stopping on test data

MODELS = {"(a) frames": (1, 0), "(b) frames + since": (1, 1), "(c) since only": (0, 1)}
data = {n: load(n) for n in FILES}
print({n: len(d[0]) for n, d in data.items()}, "samples per file")

# mae[name][fold, seed]
mae = {k: np.zeros((len(FILES), len(SEEDS))) for k in ["baseline", *MODELS]}
for f, held in enumerate(FILES):
    tr = [torch.cat([data[n][i] for n in FILES if n != held]) for i in range(3)]
    te = data[held]
    mae["baseline"][f] = (te[2] - tr[2].mean()).abs().mean().item()
    for k, (uf, us) in MODELS.items():
        for s in SEEDS:
            mae[k][f, s] = train_eval(tr, te, uf, us, s)
    print(f"test={held}: " + "  ".join(f"{k}={v[f].mean():.2f}" for k, v in mae.items()), flush=True)

print("\nMean test MAE in frames (mean over folds, then mean +- std over seeds):")
overall = {k: v.mean(axis=0) for k, v in mae.items()}        # one number per seed
for k, v in overall.items():
    print(f"  {k:20s} {v.mean():.2f} +- {v.std():.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, (k, v) in enumerate(mae.items()):
    ax[0].bar(np.arange(4) + 0.2 * j, v.mean(axis=1), 0.2, yerr=v.std(axis=1), label=k)
ax[0].set_xticks(np.arange(4) + 0.3); ax[0].set_xticklabels(FILES)
ax[0].set(ylabel="test MAE (frames)", title="per test file (error bar = std over seeds)"); ax[0].legend()
ax[1].bar(list(overall), [v.mean() for v in overall.values()], yerr=[v.std() for v in overall.values()])
ax[1].set(ylabel="test MAE (frames)", title="mean over test files"); ax[1].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig("results/v1/probe.png", dpi=110)
