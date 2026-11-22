# Pretrain the flow-matching head on true frames, before any reward (run on a GPU node, from the repo root).
# Usage: PYTHONPATH=scripts python scripts/pretrain_head_v2.py --twalls 90 94 ...
#
# Each training pair is one 5-frame window of a training file:
#   condition = backbone prediction of the next 5 frames, the last input frame, and the wall temperature;
#   target    = residual = true next 5 frames - backbone prediction, on the 128 x 128 grid.
# The backbone predictions come from prepare_head_data_v2.py; the true frames come from Twall_<T>_fields128.npy.
# The last 10% of the windows of each file are held out, to measure a validation loss after each epoch.
# Saves <OUT>/head_pretrained.pt: the head weights, its width, the normalization values and the training files.
import argparse, json, os, time
import numpy as np, torch
from flow_matching_head import ResidualHead, Normalizer, flow_matching_loss

parser = argparse.ArgumentParser()
parser.add_argument("--twalls", type=int, nargs="+", required=True)
parser.add_argument("--data", default="/pub/ynagaraj/bubbleml2/PoolBoiling-Subcooled-FC72-2D")
parser.add_argument("--head-data", default="/pub/ynagaraj/bubble_posttrain/head_data")
parser.add_argument("--out", default="/pub/ynagaraj/bubble_posttrain")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--width", type=int, default=64, help="channels of the first U-Net level")
args = parser.parse_args()

FRAMES, START_FRAME = 5, 100
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
normalizer = Normalizer(json.load(open(f"{args.head_data}/scales.json"))).to(device)

# Load every window into CPU memory as float16.
predictions, truths, last_inputs, wall_temps, is_validation = [], [], [], [], []
for t in args.twalls:
    prediction = np.load(f"{args.head_data}/Twall_{t}_pred128.npy")              # (windows, 5, 4, 128, 128)
    fields = np.load(f"{args.data}/Twall_{t}_fields128.npy", mmap_mode="r")
    first = START_FRAME + FRAMES * np.arange(len(prediction))                      # first input frame of each window
    predictions.append(prediction)
    truths.append(np.stack([fields[f + FRAMES:f + 2 * FRAMES] for f in first]))
    last_inputs.append(np.stack([fields[f + FRAMES - 1] for f in first]))
    wall_temps.append(np.full(len(prediction), float(t)))
    is_validation.append(np.arange(len(prediction)) >= int(0.9 * len(prediction)))
predictions, truths, last_inputs = (torch.from_numpy(np.concatenate(a)) for a in (predictions, truths, last_inputs))
wall_temps = torch.from_numpy(np.concatenate(wall_temps)).float()
is_validation = torch.from_numpy(np.concatenate(is_validation))
train_windows, validation_windows = torch.nonzero(~is_validation)[:, 0], torch.nonzero(is_validation)[:, 0]
print(f"{len(train_windows)} training and {len(validation_windows)} validation windows", flush=True)


def make_batch(windows):
    """Normalized residuals and conditions for the given window indices, on the GPU."""
    prediction, truth, last_input = (a[windows].to(device).float() for a in (predictions, truths, last_inputs))
    condition = normalizer.build_condition(prediction, last_input, wall_temps[windows].to(device))
    return normalizer.to_normalized_residual(truth - prediction), condition


head = ResidualHead(args.width).to(device)
optimizer = torch.optim.AdamW(head.parameters(), args.lr, weight_decay=1e-4)
total_steps = args.epochs * (len(train_windows) // args.batch)
schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=total_steps, pct_start=0.05)
print(f"head: {sum(p.numel() for p in head.parameters()) / 1e6:.2f}M parameters", flush=True)
t0 = time.time()
for epoch in range(args.epochs):
    head.train()
    train_losses = []
    for positions in torch.randperm(len(train_windows)).split(args.batch):
        if len(positions) < args.batch:
            continue
        loss = flow_matching_loss(head, *make_batch(train_windows[positions]))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step(); schedule.step()
        train_losses.append(loss.item())
    head.eval()
    torch.manual_seed(1)   # the same random noise for every validation pass, so the losses are comparable
    with torch.no_grad():
        validation_loss = np.mean([flow_matching_loss(head, *make_batch(windows)).item()
                                   for windows in validation_windows.split(args.batch)])
    print(f"epoch {epoch}: train loss {np.mean(train_losses):.4f}, validation loss {validation_loss:.4f}, "
          f"{time.time() - t0:.0f} s", flush=True)
    torch.manual_seed(epoch + 2)

os.makedirs(args.out, exist_ok=True)
torch.save({"head": head.state_dict(), "width": args.width, "scales": normalizer.values, "twalls": args.twalls},
           f"{args.out}/head_pretrained.pt")
print("saved", f"{args.out}/head_pretrained.pt")
