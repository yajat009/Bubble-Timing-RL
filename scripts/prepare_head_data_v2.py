# Prepare the training data for the flow-matching head (run on a GPU node, from the repo root).
# Usage: PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts python scripts/prepare_head_data_v2.py --ckpt <S ckpt> --twalls 90 94 ...
#
# For each training file (needs Twall_<T>_fields512.npy from download_fullres_v2.py):
#   1. Run the frozen backbone on every 5-frame window that starts at frame 100, 105, 110, ...
#   2. Save its prediction averaged to 128 x 128: <OUT>/Twall_<T>_pred128.npy, float16, (windows, 5, 4, 128, 128).
#      The true next frames and the last input frame come from Twall_<T>_fields128.npy, so they are not saved again.
#   3. Check the coarse departure detector against the full-resolution departure files.
# Then save <OUT>/scales.json: the field and residual scales that the head uses (from all the files above).
import argparse, json, os
import numpy as np, torch
from fm_head import pool128
from posttrain_common import DATA, FRAME_DT, FIRST, Backbone, load_meta
from bubbleformer_io import fluid_params
from detect_departures import find_departures

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--twalls", type=int, nargs="+", required=True)
p.add_argument("--data", default=DATA)
p.add_argument("--out", default="/pub/ynagaraj/bubble_posttrain/head_data")
p.add_argument("--batch", type=int, default=4)
p.add_argument("--max-windows", type=int, default=0, help="use only the first N windows per file (for a quick test)")
args = p.parse_args()

W, START = 5, 100
dev = "cuda" if torch.cuda.is_available() else "cpu"
backbone = Backbone(args.ckpt, dev)
os.makedirs(args.out, exist_ok=True)
field_sum, field_sq, res_sq, n_field, n_res = 0, 0, 0, 0, 0

for t in args.twalls:
    meta = load_meta(t, args.data)
    full = np.load(f"{args.data}/Twall_{t}_fields512.npy", mmap_mode="r")
    f128 = np.load(f"{args.data}/Twall_{t}_fields128.npy", mmap_mode="r")
    starts = np.arange(START, len(full) - 2 * W + 1, W)
    if args.max_windows:
        starts = starts[:args.max_windows]
    fluid = fluid_params(meta).to(dev)
    path = f"{args.out}/Twall_{t}_pred128.npy"
    if not os.path.exists(path):
        pred128 = np.zeros((len(starts), W, 4, 128, 128), np.float16)
        for i in range(0, len(starts), args.batch):
            s = starts[i:i + args.batch]
            x = torch.from_numpy(np.stack([full[k:k + W] for k in s]).astype(np.float32)).to(dev)
            pred128[i:i + len(s)] = pool128(backbone(x, fluid.expand(len(s), -1))).cpu().numpy()
        np.save(path, pred128)
    pred128 = np.load(path).astype(np.float32)
    truth128 = np.stack([f128[k + W:k + 2 * W] for k in starts]).astype(np.float32)
    field_sum = field_sum + truth128.sum(axis=(0, 1, 3, 4)); field_sq = field_sq + (truth128 ** 2).sum(axis=(0, 1, 3, 4))
    n_field += truth128.size // 4
    res_sq = res_sq + ((truth128 - pred128) ** 2).sum(axis=(0, 1, 3, 4)); n_res += truth128.size // 4
    # Check the coarse (128 x 128) departure detector against the full-resolution one.
    v = np.load(f"{args.data}/Twall_{t}_vapor.npy", mmap_mode="r")
    vapor128 = np.stack([v[k].reshape(128, 4, 128, 4).mean(axis=(1, 3)) > 0.5 for k in range(FIRST, len(v))])
    coarse = len({k for k, _ in find_departures(vapor128, FRAME_DT, cell=4)[0]})
    fine = len(np.load(f"results/v2/departure_frames_Twall_{t}.npy"))
    print(f"Twall {t}: {len(starts)} windows; departures, coarse detector {coarse}, full-resolution detector {fine}", flush=True)

mean = field_sum / n_field
twalls = np.array(args.twalls, float)
# A floor on each scale avoids a division by zero for a field that does not vary.
scales = {"field_mean": mean.tolist(), "field_std": np.maximum(np.sqrt(field_sq / n_field - mean ** 2), 1e-6).tolist(),
          "residual_std": np.maximum(np.sqrt(res_sq / n_res), 1e-6).tolist(), "twall_mean": float(twalls.mean()),
          "twall_std": float(max(twalls.std(), 1.0)), "twalls": args.twalls}
json.dump(scales, open(f"{args.out}/scales.json", "w"), indent=1)
print("scales", json.dumps(scales))
