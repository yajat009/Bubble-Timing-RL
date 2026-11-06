# Median baseline for the T3 departure probe (scripts/departure_probe_v2.py). Run from the repo root.
# Why: the probe models are trained with absolute error. A model that learns nothing useful
# predicts the median target, not the mean. So the fair "no information" baseline is the
# training median. The probe reports only the mean baseline; this script adds the median.
# The windows, training stride, and irregular-interval rule are the same as in the probe.
import numpy as np

TRAIN_TWALLS = [86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116]
TEST_TWALL, WINDOW_FRAMES, START_FRAME, N_FRAMES, TRAIN_STRIDE = 97, 5, 100, 2001, 5

def window_targets(twall):
    """Return, for each window: frames until the next departure, and the length of the interval it is in."""
    departures = np.load(f"results/v2/departure_frames_Twall_{twall}.npy")
    time_until, intervals = [], []
    for i in range(START_FRAME + WINDOW_FRAMES - 1, N_FRAMES):
        past, future = departures[departures <= i], departures[departures > i]
        if len(past) and len(future):
            time_until.append(future.min() - i); intervals.append(future.min() - past.max())
    return np.array(time_until, float), np.array(intervals, float)

train_target = np.concatenate([window_targets(twall)[0][::TRAIN_STRIDE] for twall in TRAIN_TWALLS])
test_target, test_interval = window_targets(TEST_TWALL)
train_intervals = np.concatenate([np.diff(np.load(f"results/v2/departure_frames_Twall_{twall}.npy")) for twall in TRAIN_TWALLS])
irregular = np.abs(test_interval - train_intervals.mean()) > train_intervals.std(ddof=1)

for name, guess in (("mean", train_target.mean()), ("median", np.median(train_target))):
    error = np.abs(test_target - guess)
    print(f"training {name} = {guess:.2f}: test MAE all {error.mean():.2f}, irregular {error[irregular].mean():.2f}")
print(f"best constant on the test file itself (its median, an unfair lower bound): "
      f"MAE {np.abs(test_target - np.median(test_target)).mean():.2f}")
