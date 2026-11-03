import sys, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "/pub/ynagaraj/bubbleml/PoolBoiling-SubCooled-FC72-2D"  # dir with *_vapor.npy
FILES = ["Twall-100", "Twall-103", "Twall-106", "Twall-110"]
W = 5                      # frames per window
EPOCHS, LR, BS = 40, 1e-3, 32
dev = "cuda"

def load(name):
    v = np.load(f"{D}/{name}_vapor.npy").astype(np.float32)        # (T,384,384) vapor mask
    T = len(v)
    v = v.reshape(T, 96, 4, 96, 4).mean(axis=(2, 4))               # downsample 4x
    deps = np.load(f"departures_{name}.npy")
    X, since, until = [], [], []
    for t in range(W - 1, T):                                      # window = frames t-W+1..t
        past, future = deps[deps <= t], deps[deps > t]
        if len(past) == 0 or len(future) == 0:                     # need both so (a) and (b) share samples
            continue
        X.append(v[t - W + 1:t + 1]); since.append(t - past.max()); until.append(future.min() - t)
    return [torch.tensor(np.array(a), dtype=torch.float32) for a in (X, since, until)]

class CNN(nn.Module):
    def __init__(self, use_since):
        super().__init__()
        self.use_since = use_since
        self.conv = nn.Sequential(
            nn.Conv2d(W, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 + use_since, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, since):
        h = self.conv(x)
        if self.use_since:
            h = torch.cat([h, since[:, None] / 10], 1)
        return self.head(h)[:, 0] * 10                              # output in frames

def train_eval(tr, te, use_since, seed):
    torch.manual_seed(seed)
    m = CNN(use_since).to(dev)
    opt = torch.optim.Adam(m.parameters(), LR)
    X, s, y = [a.to(dev) for a in tr]
    for _ in range(EPOCHS):
        for i in torch.randperm(len(X), device=dev).split(BS):
            loss = (m(X[i], s[i]) - y[i]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        Xt, st, yt = [a.to(dev) for a in te]
        pred = m(Xt, st)
    return (pred - yt).abs().mean().item(), pred.cpu().numpy()

data = {n: load(n) for n in FILES}
print({n: len(d[0]) for n, d in data.items()}, "samples per file")
res = {"baseline": [], "(a) frames": [], "(b) frames+since": []}
preds = {"(a) frames": [], "(b) frames+since": []}; truth = []
for held in FILES:
    tr = [torch.cat([data[n][i] for n in FILES if n != held]) for i in range(3)]
    te = data[held]
    base = (te[2] - tr[2].mean()).abs().mean().item()
    a, pa = train_eval(tr, te, False, 0)
    b, pb = train_eval(tr, te, True, 0)
    print(f"test={held}: baseline MAE={base:.2f}  (a) frames={a:.2f}  (b) frames+since={b:.2f}")
    res["baseline"].append(base); res["(a) frames"].append(a); res["(b) frames+since"].append(b)
    preds["(a) frames"].append(pa); preds["(b) frames+since"].append(pb); truth.append(te[2].numpy())
print("mean over folds:", {k: round(float(np.mean(v)), 2) for k, v in res.items()}, "(frames)")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for j, (k, v) in enumerate(res.items()):
    ax[0].bar(np.arange(4) + 0.27 * j, v, 0.27, label=k)
ax[0].set_xticks(np.arange(4) + 0.27); ax[0].set_xticklabels(FILES); ax[0].set(ylabel="test MAE (frames)", title="leave-one-file-out"); ax[0].legend()
y = np.concatenate(truth)
for a_, k in zip(ax[1:], preds):
    a_.scatter(y, np.concatenate(preds[k]), s=6, alpha=0.4); a_.plot([0, y.max()], [0, y.max()], "k--")
    a_.set(title=k, xlabel="true frames to next departure", ylabel="predicted")
plt.tight_layout(); plt.savefig("probe.png", dpi=110)
