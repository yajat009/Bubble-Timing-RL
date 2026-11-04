# Median baseline for the T3 probe (scripts/probe_v2.py). Run from the repo root.
# Why: the probe models are trained with absolute error. A model that learns nothing useful
# predicts the median target, not the mean. So the fair "no information" baseline is the
# training median. probe_v2.py reports only the mean baseline; this script adds the median.
# The windows, training stride, and irregular-interval rule are the same as in probe_v2.py.
import numpy as np

TRAIN = [86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116]
TEST, W, START, T, TRAIN_STRIDE = 97, 5, 100, 2001, 5

def targets(t):
    deps = np.load(f"results/v2/departures_Twall_{t}.npy")
    y, gap = [], []
    for i in range(START + W - 1, T):
        past, future = deps[deps <= i], deps[deps > i]
        if len(past) and len(future):
            y.append(future.min() - i); gap.append(future.min() - past.max())
    return np.array(y, float), np.array(gap, float)

y_tr = np.concatenate([targets(t)[0][::TRAIN_STRIDE] for t in TRAIN])
y_te, gap_te = targets(TEST)
gaps = np.concatenate([np.diff(np.load(f"results/v2/departures_Twall_{t}.npy")) for t in TRAIN])
irregular = np.abs(gap_te - gaps.mean()) > gaps.std(ddof=1)

for name, c in (("mean", y_tr.mean()), ("median", np.median(y_tr))):
    e = np.abs(y_te - c)
    print(f"training {name} = {c:.2f}: test MAE all {e.mean():.2f}, irregular {e[irregular].mean():.2f}")
print(f"best constant on the test file itself (its median, an unfair lower bound): "
      f"MAE {np.abs(y_te - np.median(y_te)).mean():.2f}")
