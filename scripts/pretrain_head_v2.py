# Pretrain the flow-matching head on true frames (step 1 of the training plan; run on a GPU node from the repo root).
# Usage: PYTHONPATH=scripts python scripts/pretrain_head_v2.py --twalls 90 94 ...
#
# Training pairs (from prepare_head_data_v2.py and the 128 x 128 simulation fields):
#   condition = backbone prediction of the next 5 frames, the last input frame, and the wall temperature;
#   target    = residual = true next 5 frames - backbone prediction, on the 128 x 128 grid.
# The last 10% of the windows of each file are held out to measure the validation loss.
# Saves <OUT>/head_pretrained.pt with the head weights and the scales.
import argparse, json, os, time
import numpy as np, torch
from fm_head import Head, Scales, cfm_loss

p = argparse.ArgumentParser()
p.add_argument("--twalls", type=int, nargs="+", required=True)
p.add_argument("--data", default="/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D")
p.add_argument("--head-data", default="/pub/ynagaraj/bubble_posttrain/head_data")
p.add_argument("--out", default="/pub/ynagaraj/bubble_posttrain")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--width", type=int, default=64)
args = p.parse_args()

W, START = 5, 100
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
scales = Scales(json.load(open(f"{args.head_data}/scales.json"))).to(dev)

# Load every window into CPU memory as float16: pred (N, 5, 4, 128, 128), truth (same), last (N, 4, 128, 128), twall (N,).
pred, truth, last, twall, val = [], [], [], [], []
for t in args.twalls:
    p128 = np.load(f"{args.head_data}/Twall_{t}_pred128.npy")
    f128 = np.load(f"{args.data}/Twall_{t}_fields128.npy", mmap_mode="r")
    starts = START + W * np.arange(len(p128))
    pred.append(p128); truth.append(np.stack([f128[k + W:k + 2 * W] for k in starts]))
    last.append(np.stack([f128[k + W - 1] for k in starts])); twall.append(np.full(len(p128), float(t)))
    val.append(np.arange(len(p128)) >= int(0.9 * len(p128)))
pred, truth, last = (torch.from_numpy(np.concatenate(a)) for a in (pred, truth, last))
twall, val = torch.from_numpy(np.concatenate(twall)).float(), torch.from_numpy(np.concatenate(val))
train_idx, val_idx = torch.nonzero(~val)[:, 0], torch.nonzero(val)[:, 0]
print(f"{len(train_idx)} training and {len(val_idx)} validation windows", flush=True)


def batch(idx):
    """Normalized residuals x and conditions c for the windows idx, on the GPU."""
    pr, tr, la = (a[idx].to(dev).float() for a in (pred, truth, last))
    return scales.from_residual(tr - pr), scales.condition(pr, la, twall[idx].to(dev))


head = Head(args.width).to(dev)
opt = torch.optim.AdamW(head.parameters(), args.lr, weight_decay=1e-4)
steps = args.epochs * (len(train_idx) // args.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
print(f"head: {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M parameters", flush=True)
t0 = time.time()
for epoch in range(args.epochs):
    head.train()
    losses = []
    for b in torch.randperm(len(train_idx)).split(args.batch):
        if len(b) < args.batch:
            continue
        loss = cfm_loss(head, *batch(train_idx[b]))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
        losses.append(loss.item())
    head.eval()
    torch.manual_seed(1)   # the same noise every epoch, so validation losses are comparable
    with torch.no_grad():
        v = np.mean([cfm_loss(head, *batch(b)).item() for b in val_idx.split(args.batch)])
    print(f"epoch {epoch}: train {np.mean(losses):.4f}, validation {v:.4f}, {time.time() - t0:.0f} s", flush=True)
    torch.manual_seed(epoch + 2)

os.makedirs(args.out, exist_ok=True)
torch.save({"head": head.state_dict(), "width": args.width, "scales": scales.d, "twalls": args.twalls},
           f"{args.out}/head_pretrained.pt")
print("saved", f"{args.out}/head_pretrained.pt")
