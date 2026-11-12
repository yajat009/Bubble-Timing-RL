# Is our Bubbleformer setup correct? Check its short-term accuracy before trusting the T6 rollout.
# The T6 rollout already differs from the simulation in its first 300 frames. That can be a model failure,
# or a setup error (wrong conditioning values, normalization, or field order). A correct setup gives
# short-term errors near the values in the Bubbleformer paper, and much lower than a "nothing changes" guess.
#
# Usage (from the repo root, on a GPU node):
#   PYTHONPATH=/pub/ynagaraj/Bubbleformer python scripts/accuracy_check_v2.py \
#       --ckpt <path>/Bubbleformer-S-PB-Subcooled.ckpt --twall 97
#
# Steps:
#   1. Take N start frames, evenly spaced after frame 100.
#   2. From each start, give the model 5 simulation frames and let it predict 2 bundles (10 frames):
#      bundle 1 from the simulation frames, bundle 2 from its own bundle 1, as in a rollout.
#   3. Compare each predicted frame with the simulation: relative L2 error per field, the metric of
#      Bubbleformer's LpLoss (||pred - true|| / ||true|| over the 512 x 512 frame).
#   4. Compare with persistence: always repeat the last input frame. A model that does not beat
#      persistence at frame 1 is almost surely set up wrong.
#   5. Also report how well the predicted vapor region matches (intersection over union of dfun > 0).
# The 15 frames per start are read from Hugging Face if Twall_<T>.hdf5 is not in DATA (about 60 MB per start).
import argparse, json, os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bubbleformer_io import FIELDS, load_model, fluid_params, read_frames

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--twall", type=int, default=97)
p.add_argument("--starts", type=int, default=20, help="number of start frames")
p.add_argument("--data", default="/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D")
p.add_argument("--out", default="results/v2")
args = p.parse_args()

W, START, LAST = 5, 100, 2000          # Bubbleformer's window and start frame; last frame in a file
BUNDLES = 2                            # 2 bundles = 10 predicted frames
name = f"Twall_{args.twall}"
meta = json.load(open(f"{args.data}/{name}.json"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

model, diff, div = load_model(args.ckpt, dev)
shift = torch.tensor([diff[k] for k in FIELDS]).view(1, 4, 1, 1)
scale = torch.tensor([div[k] for k in FIELDS]).view(1, 4, 1, 1)
cond = fluid_params(meta).to(dev)


def rel_l2(pred, true):
    """Relative L2 error per frame and field, shape (frames, 4)."""
    return (torch.linalg.vector_norm(pred - true, dim=(-2, -1)) / torch.linalg.vector_norm(true, dim=(-2, -1))).numpy()


def iou(pred, true):
    """Intersection over union of the vapor region (dfun > 0), one value per frame."""
    a, b = pred[:, 0] > 0, true[:, 0] > 0
    return ((a & b).sum(dim=(-2, -1)) / (a | b).sum(dim=(-2, -1)).clamp(min=1)).numpy()


starts = np.linspace(START, LAST - W * (BUNDLES + 1), args.starts).astype(int)
err_model, err_persist, iou_model, iou_persist = [], [], [], []
for i, s in enumerate(starts):
    frames = read_frames(args.data, name, s, s + W * (BUNDLES + 1))      # (15, 4, 512, 512), physical units
    inp, true = frames[:W], frames[W:]
    x = ((inp - shift) / scale)[None].to(dev)
    preds = []
    with torch.no_grad():
        for _ in range(BUNDLES):
            x = model(x, cond)                                           # each prediction is the next input
            preds.append((x[0].cpu() * scale + shift))
    pred = torch.cat(preds)
    persist = inp[-1:].expand_as(true)                                   # repeat the last input frame
    err_model.append(rel_l2(pred, true)); err_persist.append(rel_l2(persist, true))
    iou_model.append(iou(pred, true)); iou_persist.append(iou(persist, true))
    print(f"start {s} ({i + 1} of {len(starts)}): frame-1 rel L2 model {err_model[-1][0].round(3)}, "
          f"persistence {err_persist[-1][0].round(3)}", flush=True)

err_model, err_persist = np.array(err_model), np.array(err_persist)      # (starts, 10, 4)
iou_model, iou_persist = np.array(iou_model), np.array(iou_persist)      # (starts, 10)

# Table: mean over starts, for frame 1, frames 1-5 (bundle 1), and frames 6-10 (bundle 2).
spans = {"frame 1": slice(0, 1), "frames 1-5": slice(0, 5), "frames 6-10": slice(5, 10)}
lines = [f"# Bubbleformer short-term accuracy, Twall {args.twall}", "",
         f"{len(starts)} start frames from {starts[0]} to {starts[-1]}. Checkpoint: `{os.path.basename(args.ckpt)}`. "
         "Relative L2 error (lower is better) and vapor IoU (higher is better), mean over starts. "
         "Persistence = repeat the last input frame.", "",
         "| Span | Predictor | " + " | ".join(f"rel L2 {f}" for f in FIELDS) + " | vapor IoU |",
         "|---" * (len(FIELDS) + 3) + "|"]
for span, sl in spans.items():
    for label, e, u in (("Bubbleformer", err_model, iou_model), ("persistence", err_persist, iou_persist)):
        vals = e[:, sl].mean(axis=(0, 1))
        lines.append(f"| {span} | {label} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {u[:, sl].mean():.3f} |")
table = "\n".join(lines)
print(table)
os.makedirs(args.out, exist_ok=True)
out = f"{args.out}/accuracy_{name}"
open(f"{out}.md", "w").write(table + "\n")
json.dump({"starts": starts.tolist(), "fields": FIELDS,
           "rel_l2_model": err_model.tolist(), "rel_l2_persistence": err_persist.tolist(),
           "iou_model": iou_model.tolist(), "iou_persistence": iou_persist.tolist()}, open(f"{out}.json", "w"))

# Figure: error against lead frame, one panel per field, model and persistence (mean and spread over starts).
lead = np.arange(1, W * BUNDLES + 1)
fig, ax = plt.subplots(1, 5, figsize=(20, 3.5))
for k, f in enumerate(FIELDS + ["vapor IoU"]):
    for label, e in (("Bubbleformer", err_model if k < 4 else iou_model),
                     ("persistence", err_persist if k < 4 else iou_persist)):
        v = e[..., k] if k < 4 else e
        ax[k].plot(lead, v.mean(axis=0), "o-", label=label)
        ax[k].fill_between(lead, np.percentile(v, 10, axis=0), np.percentile(v, 90, axis=0), alpha=0.2)
    ax[k].set(title=f if k == 4 else f"rel L2, {f}", xlabel="frames ahead")
ax[0].legend()
plt.tight_layout(); plt.savefig(f"{out}.png", dpi=110)
