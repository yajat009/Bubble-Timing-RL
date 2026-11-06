# T3 departure probe on BubbleML 2.0 (see analysis/TEST_PLAN.md).
# Question: do Bubbleformer's 5 input frames contain the time to the next bubble departure?
# Run from the repo root: python scripts/departure_probe_v2.py
#
# Data: the 4 Bubbleformer fields (dfun, temperature, velx, vely) at 128 x 128, from scripts/download_bubbleml_v2.py.
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

DATA_DIR = "/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D"
TRAIN_TWALLS = [86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116]
TEST_TWALLS = [97]
WINDOW_FRAMES = 5   # frames per input window
START_FRAME = 100   # Bubbleformer skips the first 100 frames
SEEDS = range(5)
EPOCHS, LEARNING_RATE, BATCH_SIZE = 10, 1e-3, 64
TRAIN_STRIDE = 5    # use every 5th training window; neighbours share 4 of 5 frames. Test windows use every window.
SCALE = 10.0        # targets and the time input are divided by this, so the network works with numbers near 1
device = "cuda"

def load_file(twall):
    """Load one file. Return its frames, and for each window: last frame, time since, time until, interval length."""
    frames = torch.from_numpy(np.load(f"{DATA_DIR}/Twall_{twall}_fields128.npy"))   # (T, 4, 128, 128) float16
    departures = np.load(f"results/v2/departure_frames_Twall_{twall}.npy")
    last_frames, time_since, time_until, intervals = [], [], [], []
    for i in range(START_FRAME + WINDOW_FRAMES - 1, len(frames)):          # window = frames i-4 ... i
        past, future = departures[departures <= i], departures[departures > i]
        if len(past) == 0 or len(future) == 0:                             # need both, so all models share samples
            continue
        last_frames.append(i); time_since.append(i - past.max()); time_until.append(future.min() - i)
        intervals.append(future.min() - past.max())                        # length of the interval this window is in
    return (frames, np.array(last_frames), np.array(time_since, np.float32),
            np.array(time_until, np.float32), np.array(intervals, np.float32))

data = {twall: load_file(twall) for twall in TRAIN_TWALLS + TEST_TWALLS}
print({twall: len(d[1]) for twall, d in data.items()}, "windows per file", flush=True)

# Per-field mean and std from the training files only.
sample = torch.cat([data[twall][0][START_FRAME::20].float() for twall in TRAIN_TWALLS])
FIELD_MEAN = sample.mean(dim=(0, 2, 3)).view(1, 1, 4, 1, 1).to(device)
FIELD_STD = sample.std(dim=(0, 2, 3)).view(1, 1, 4, 1, 1).to(device)
# All files in one GPU tensor (about 260 MB per file in float16). FILE_OFFSET[twall] is where that file starts.
FILE_OFFSET, n_frames = {}, 0
for twall in data:
    FILE_OFFSET[twall] = n_frames; n_frames += len(data[twall][0])
ALL_FRAMES = torch.cat([data[twall][0] for twall in data]).to(device)
WINDOW_STEPS = torch.arange(-WINDOW_FRAMES + 1, 1, device=device)

def collect_windows(twalls, stride=1):
    """Return, for every window: its last frame's index in ALL_FRAMES, time since, time until, and interval length."""
    last_frames = np.concatenate([FILE_OFFSET[twall] + data[twall][1][::stride] for twall in twalls])
    column = lambda k: np.concatenate([data[twall][k][::stride] for twall in twalls])
    return last_frames, column(2), column(3), column(4)

def gather_inputs(last_frames):
    """Gather frames i-4 ... i for each window and stack them as 20 channels (5 frames x 4 fields)."""
    x = ALL_FRAMES[torch.as_tensor(last_frames, device=device)[:, None] + WINDOW_STEPS].float()
    return ((x - FIELD_MEAN) / FIELD_STD).flatten(1, 2)

class DeparturePredictor(nn.Module):
    def __init__(self, use_frames, use_time):
        super().__init__()
        self.use_frames, self.use_time = use_frames, use_time
        self.conv = nn.Sequential(
            nn.Conv2d(4 * WINDOW_FRAMES, 32, 5, 2, 2), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 * use_frames + use_time, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x, time_since):
        parts = []
        if self.use_frames: parts.append(self.conv(x))
        if self.use_time: parts.append(time_since[:, None] / SCALE)
        return self.head(torch.cat(parts, 1))[:, 0] * SCALE

def predict(model, last_frames, time_since):
    out = []
    with torch.no_grad():
        for start in range(0, len(last_frames), 256):
            x = gather_inputs(last_frames[start:start + 256]) if model.use_frames else None
            out.append(model(x, torch.from_numpy(time_since[start:start + 256]).to(device)).cpu())
    return torch.cat(out).numpy()

def train_and_predict(train_set, test_set, use_frames, use_time, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DeparturePredictor(use_frames, use_time).to(device)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    last_frames, time_since, time_until, _ = train_set
    for _ in range(EPOCHS):
        for batch in np.array_split(np.random.permutation(len(last_frames)), len(last_frames) // BATCH_SIZE):
            x = gather_inputs(last_frames[batch]) if use_frames else None
            prediction = model(x, torch.from_numpy(time_since[batch]).to(device))
            loss = (prediction - torch.from_numpy(time_until[batch]).to(device)).abs().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    return predict(model, test_set[0], test_set[1])                        # no early stopping on test data

train_set, test_set = collect_windows(TRAIN_TWALLS, TRAIN_STRIDE), collect_windows(TEST_TWALLS)
test_target, test_interval = test_set[2], test_set[3]
train_mean_target = train_set[2].mean()
# Irregular windows: inside an interval more than 1 std from the mean training interval.
train_intervals = np.concatenate([np.diff(np.load(f"results/v2/departure_frames_Twall_{twall}.npy")) for twall in TRAIN_TWALLS])
irregular = np.abs(test_interval - train_intervals.mean()) > train_intervals.std(ddof=1)
print(f"train windows {len(train_set[0])}, test windows {len(test_set[0])}, irregular test windows {irregular.sum()}; "
      f"train interval mean {train_intervals.mean():.1f} std {train_intervals.std(ddof=1):.1f} frames", flush=True)

MODELS = {"(b) time only": (0, 1), "(c) frames": (1, 0), "(d) frames + time": (1, 1)}
mae = {"(a) mean only": np.full(len(SEEDS), np.abs(test_target - train_mean_target).mean())}
mae_irregular = {"(a) mean only": np.full(len(SEEDS), np.abs(test_target - train_mean_target)[irregular].mean())}
for model_name, (use_frames, use_time) in MODELS.items():
    mae[model_name], mae_irregular[model_name] = np.zeros(len(SEEDS)), np.zeros(len(SEEDS))
    for seed in SEEDS:
        error = np.abs(train_and_predict(train_set, test_set, use_frames, use_time, seed) - test_target)
        mae[model_name][seed], mae_irregular[model_name][seed] = error.mean(), error[irregular].mean()
        print(f"{model_name} seed {seed}: MAE {mae[model_name][seed]:.2f}, irregular {mae_irregular[model_name][seed]:.2f}", flush=True)

print(f"\nTest MAE in frames on Twall 97 (mean +- sample std over {len(SEEDS)} seeds):")
for model_name in mae:
    print(f"  {model_name:20s} all {mae[model_name].mean():.2f} +- {mae[model_name].std(ddof=1):.2f}   "
          f"irregular {mae_irregular[model_name].mean():.2f} +- {mae_irregular[model_name].std(ddof=1):.2f}")

# Pass criteria from analysis/TEST_PLAN.md (T3). seed_std = sample std over seeds of model (c).
mean_a, mean_b, mean_c, mean_d = (mae[model_name].mean() for model_name in mae)
seed_std = mae["(c) frames"].std(ddof=1)
irregular_b, irregular_c = mae_irregular["(b) time only"].mean(), mae_irregular["(c) frames"].mean()
criteria = {"1: (c) beats (a) by more than 2 seed std": mean_a - mean_c > 2 * seed_std,
            "2: |(d) - (c)| within 1 seed std": abs(mean_d - mean_c) <= seed_std,
            "3: on irregular intervals, (c) beats (b) by more than 2 seed std": irregular_b - irregular_c > 2 * seed_std}
print(f"\nseed std of (c) = {seed_std:.3f}")
for criterion, passed in criteria.items():
    print(f"  criterion {criterion}: {'PASS' if passed else 'FAIL'}")

json.dump({"mae": {k: v.tolist() for k, v in mae.items()}, "mae_irregular": {k: v.tolist() for k, v in mae_irregular.items()},
           "seed_std_of_model_c": float(seed_std), "criteria": {k: bool(v) for k, v in criteria.items()},
           "test_windows": int(len(test_target)), "irregular_windows": int(irregular.sum())},
          open("results/v2/departure_probe_v2.json", "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for j, (results, title) in enumerate([(mae, "all test windows"), (mae_irregular, "irregular intervals only")]):
    ax[j].bar(list(results), [v.mean() for v in results.values()], yerr=[v.std(ddof=1) for v in results.values()])
    ax[j].set(ylabel="test MAE (frames)", title=f"Twall 97, {title}"); ax[j].tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig("results/v2/departure_probe_v2.png", dpi=110)
