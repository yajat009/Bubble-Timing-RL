# Prepare the training data for the flow-matching head (run on a GPU node, from the repo root).
# Usage: PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts python scripts/prepare_head_data_v2.py --ckpt <S ckpt> --twalls 90 94 ...
#
# For each training file (it needs Twall_<T>_fields512.npy from download_fullres_v2.py):
#   1. Run the frozen backbone on every 5-frame window that starts at frame 100, 105, 110, ...
#   2. Save the prediction averaged to 128 x 128 as <OUT>/Twall_<T>_pred128.npy, float16, (windows, 5, 4, 128, 128).
#      The true next frames and the last input frame are already in Twall_<T>_fields128.npy.
#   3. Compare the coarse (128 x 128) departure detector with the full-resolution departure file. The rewards use
#      the coarse detector, so its departure counts must be close to the full-resolution counts.
# Then save <OUT>/scales.json: the normalization values of the head, from all the files above.
import argparse, json, os
import numpy as np, torch
from flow_matching_head import average_to_128
from posttrain_common import DATA, FRAME_DT, FIRST_PREDICTED, Backbone, load_metadata
from bubbleformer_io import fluid_params
from detect_departures import find_departures

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="Bubbleformer-S checkpoint")
parser.add_argument("--twalls", type=int, nargs="+", required=True)
parser.add_argument("--data", default=DATA)
parser.add_argument("--out", default="/pub/ynagaraj/bubble_posttrain/head_data")
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--max-windows", type=int, default=0, help="use only the first N windows per file (for a quick test)")
args = parser.parse_args()

FRAMES, START_FRAME = 5, 100
device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = Backbone(args.ckpt, device)
os.makedirs(args.out, exist_ok=True)
field_sum, field_square_sum, residual_square_sum, n_values = 0, 0, 0, 0

for t in args.twalls:
    metadata = load_metadata(t, args.data)
    full_frames = np.load(f"{args.data}/Twall_{t}_fields512.npy", mmap_mode="r")
    fields = np.load(f"{args.data}/Twall_{t}_fields128.npy", mmap_mode="r")
    window_starts = np.arange(START_FRAME, len(full_frames) - 2 * FRAMES + 1, FRAMES)
    if args.max_windows:
        window_starts = window_starts[:args.max_windows]
    fluid = fluid_params(metadata).to(device)
    path = f"{args.out}/Twall_{t}_pred128.npy"
    if not os.path.exists(path):
        predictions = np.zeros((len(window_starts), FRAMES, 4, 128, 128), np.float16)
        for i in range(0, len(window_starts), args.batch):
            starts = window_starts[i:i + args.batch]
            inputs = torch.from_numpy(np.stack([full_frames[s:s + FRAMES] for s in starts]).astype(np.float32)).to(device)
            predictions[i:i + len(starts)] = average_to_128(backbone(inputs, fluid.expand(len(starts), -1))).cpu().numpy()
        np.save(path, predictions)
    predictions = np.load(path).astype(np.float32)
    truths = np.stack([fields[s + FRAMES:s + 2 * FRAMES] for s in window_starts]).astype(np.float32)
    field_sum = field_sum + truths.sum(axis=(0, 1, 3, 4))
    field_square_sum = field_square_sum + (truths ** 2).sum(axis=(0, 1, 3, 4))
    residual_square_sum = residual_square_sum + ((truths - predictions) ** 2).sum(axis=(0, 1, 3, 4))
    n_values += truths.size // 4

    vapor = np.load(f"{args.data}/Twall_{t}_vapor.npy", mmap_mode="r")
    vapor_128 = np.stack([vapor[k].reshape(128, 4, 128, 4).mean(axis=(1, 3)) > 0.5 for k in range(FIRST_PREDICTED, len(vapor))])
    coarse_count = len({frame for frame, _ in find_departures(vapor_128, FRAME_DT, cell=4)[0]})
    full_frames_departures = np.load(f"results/v2/departure_frames_Twall_{t}.npy")
    full_count = int((full_frames_departures >= FIRST_PREDICTED).sum())
    print(f"Twall {t}: {len(window_starts)} windows; departures from frame {FIRST_PREDICTED}: "
          f"coarse detector {coarse_count}, full-resolution detector {full_count}", flush=True)

field_mean = field_sum / n_values
walls = np.array(args.twalls, float)
# A floor on each standard deviation avoids a division by zero for a field that does not vary.
scales = {"field_mean": field_mean.tolist(),
          "field_std": np.maximum(np.sqrt(field_square_sum / n_values - field_mean ** 2), 1e-6).tolist(),
          "residual_std": np.maximum(np.sqrt(residual_square_sum / n_values), 1e-6).tolist(),
          "twall_mean": float(walls.mean()), "twall_std": float(max(walls.std(), 1.0)), "twalls": args.twalls}
json.dump(scales, open(f"{args.out}/scales.json", "w"), indent=1)
print("normalization values", json.dumps(scales))
