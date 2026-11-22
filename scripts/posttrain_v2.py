# Post-train the flow-matching head with a reward (step 2-3 of the training plan; run on a GPU node from the repo root).
# Usage: PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts python scripts/posttrain_v2.py --ckpt <S ckpt> --reward R3 --seed 0
#
# Each iteration:
#   1. Pick one training file and one start state (5 true frames at 512 x 512, from Twall_<T>_fields512.npy).
#   2. Generate a group of G rollouts from that state with the frozen copy v_old of the head.
#   3. Score each rollout with the reward (R2: snapshot statistics; R3: timing statistics; R4: both), against the
#      simulation of the same wall temperature. Each distance is divided by its noise scale (posttrain_common.reference).
#   4. Turn the rewards into weights w in [0, 1]: above the group mean -> w > 0.5, below -> w < 0.5.
#   5. Update the head v with the DiffusionNFT loss on the kept (noise, sample, condition) triples of each rollout:
#        v+ = (1 - beta) v_old + beta v,  v- = (1 + beta) v_old - beta v,  target = sample - noise,
#        L_align = (1 / beta) mean[w |v+ - target|^2 + (1 - w) |v- - target|^2],
#      plus lambda_sup x (flow-matching loss on true training pairs) and lambda_ref x |v - v_pretrained|^2.
#      A high-reward rollout pulls v toward its samples; a low-reward rollout pushes v away from them.
#   6. Move v_old toward v (exponential moving average).
# The update rule follows ReAl and DiffusionNFT, with the settings of the proposal (beta 0.1, lambda_sup 0.1, lambda_ref 0.01).
import argparse, copy, json, os, time
import numpy as np, torch
from fm_head import Head, Scales, cfm_loss
from posttrain_common import DATA, FIRST, Backbone, load_meta, rollout, stats, distances, reward, reference, REWARDS
from bubbleformer_io import fluid_params

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--reward", choices=list(REWARDS), required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--twalls", type=int, nargs="+", default=None, help="training files; default: those of the pretrained head")
p.add_argument("--iters", type=int, default=60)
p.add_argument("--group", type=int, default=8, help="rollouts per group (G)")
p.add_argument("--bundles", type=int, default=120, help="bundles of 5 frames per rollout (120 -> 600 frames)")
p.add_argument("--keep", type=int, default=24, help="bundles per rollout kept for the update")
p.add_argument("--epochs", type=int, default=2, help="passes over the kept triples per iteration")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--lambda-sup", type=float, default=0.1)
p.add_argument("--lambda-ref", type=float, default=0.01)
p.add_argument("--ema", type=float, default=0.9, help="v_old <- ema * v_old + (1 - ema) * v after each iteration")
p.add_argument("--data", default=DATA)
p.add_argument("--work", default="/pub/ynagaraj/bubble_posttrain")
args = p.parse_args()

W, START = 5, 100
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)
ck = torch.load(f"{args.work}/head_pretrained.pt", map_location="cpu")
scales = Scales(ck["scales"]).to(dev)
twalls = args.twalls or ck["twalls"]
head = Head(ck["width"]).to(dev); head.load_state_dict(ck["head"])
head_ref = copy.deepcopy(head).eval().requires_grad_(False)       # pretrained head, for the lambda_ref term
head_old = copy.deepcopy(head).eval().requires_grad_(False)       # v_old: generates the rollouts
opt = torch.optim.AdamW(head.parameters(), args.lr, weight_decay=0.0)
backbone = Backbone(args.ckpt, dev)
n_frames = W * args.bundles

print("reference statistics ...", flush=True)
refs = {t: reference(t, n_frames, args.data) for t in twalls}
for t, r in refs.items():
    print(f"Twall {t}: rate {r['stats']['rate']:.2f} per 100 frames, noise scales "
          + ", ".join(f"{k} {v:.3g}" for k, v in r["noise"].items()), flush=True)

# Supervised pairs for the lambda_sup term (same data as the pretraining).
sup = []
for t in twalls:
    p128 = np.load(f"{args.work}/head_data/Twall_{t}_pred128.npy", mmap_mode="r")
    sup.append((t, p128, np.load(f"{args.data}/Twall_{t}_fields128.npy", mmap_mode="r")))


def sup_batch(n):
    t, p128, f128 = sup[np.random.randint(len(sup))]
    i = np.random.randint(0, len(p128), n)
    k = START + W * i
    pr = torch.from_numpy(p128[i].astype(np.float32)).to(dev)
    tr = torch.from_numpy(np.stack([f128[j + W:j + 2 * W] for j in k]).astype(np.float32)).to(dev)
    la = torch.from_numpy(np.stack([f128[j + W - 1] for j in k]).astype(np.float32)).to(dev)
    return scales.from_residual(tr - pr), scales.condition(pr, la, torch.full((n,), float(t), device=dev))


full = {t: np.load(f"{args.data}/Twall_{t}_fields512.npy", mmap_mode="r") for t in twalls}
out_dir = f"{args.work}/runs/{args.reward}_seed{args.seed}"
os.makedirs(out_dir, exist_ok=True)
log = open(f"{out_dir}/log.jsonl", "a")
t0 = time.time()
for it in range(args.iters):
    # 1-2. One start state, G rollouts from it with v_old.
    t = int(np.random.choice(twalls))
    meta = load_meta(t, args.data)
    s = int(np.random.randint(START, len(full[t]) - n_frames - W))
    x = torch.from_numpy(full[t][s:s + W].astype(np.float32)).to(dev)[None].expand(args.group, -1, -1, -1, -1).contiguous()
    fluid = fluid_params(meta).to(dev).expand(args.group, -1)
    tw = torch.full((args.group,), float(t), device=dev)
    try:
        ro = rollout(backbone, head_old, scales, x, fluid, tw, meta, args.bundles, keep=args.keep)
    except FloatingPointError as e:
        print(f"iteration {it}: {e}; skipped", flush=True)
        continue
    # 3. Rewards.
    ref = refs[t]
    st = [stats(ro["q"][g], ro["vapor128"][g], ref["threshold"]) for g in range(args.group)]
    dist = [distances(sg, ref["stats"]) for sg in st]
    rew = np.array([reward(args.reward, d, ref["noise"]) for d in dist])
    # 4. Weights from the group-normalized rewards.
    adv = (rew - rew.mean()) / (rew.std() + 1e-8)
    w = np.clip(0.5 + 0.25 * adv, 0.0, 1.0)
    # 5. DiffusionNFT update on the kept triples. ro["transitions"] holds, per kept bundle, (z0, z, c) for all G rollouts.
    z0 = torch.cat([tr[0] for tr in ro["transitions"]]); z1 = torch.cat([tr[1] for tr in ro["transitions"]])
    cc = torch.cat([tr[2] for tr in ro["transitions"]])
    ww = torch.from_numpy(np.tile(w, len(ro["transitions"]))).float()
    head.train()
    losses = []
    for _ in range(args.epochs):
        for b in torch.randperm(len(z0)).split(args.batch):
            a0, a1, c, wb = z0[b].to(dev).float(), z1[b].to(dev).float(), cc[b].to(dev).float(), ww[b].to(dev)
            tau = torch.rand(len(b), device=dev)
            zt = (1 - tau.view(-1, 1, 1, 1)) * a0 + tau.view(-1, 1, 1, 1) * a1
            target = a1 - a0
            v = head(zt, tau, c)
            with torch.no_grad():
                v_old, v_ref = head_old(zt, tau, c), head_ref(zt, tau, c)
            l_pos = ((1 - args.beta) * v_old + args.beta * v - target).pow(2).mean(dim=(1, 2, 3))
            l_neg = ((1 + args.beta) * v_old - args.beta * v - target).pow(2).mean(dim=(1, 2, 3))
            l_align = (wb * l_pos + (1 - wb) * l_neg).mean() / args.beta
            loss = l_align + args.lambda_sup * cfm_loss(head, *sup_batch(args.batch)) \
                + args.lambda_ref * (v - v_ref).pow(2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
            losses.append(loss.item())
    # 6. v_old follows v.
    with torch.no_grad():
        for po, pn in zip(head_old.parameters(), head.parameters()):
            po.mul_(args.ema).add_((1 - args.ema) * pn)
    row = {"iter": it, "twall": t, "start": s, "reward_mean": float(rew.mean()), "reward_std": float(rew.std()),
           "loss": float(np.mean(losses)), "time_s": round(time.time() - t0)}
    for k in ("rate", "wait", "dwell", "hist", "vapor"):
        row[k] = float(np.mean([d[k] for d in dist]))
    row["rollout_rate"] = float(np.mean([sg["rate"] for sg in st]))
    row["rate_ref"] = ref["stats"]["rate"]
    log.write(json.dumps(row) + "\n"); log.flush()
    print(json.dumps(row), flush=True)
    if (it + 1) % 10 == 0 or it + 1 == args.iters:
        torch.save({"head": head.state_dict(), "width": ck["width"], "scales": ck["scales"], "twalls": twalls,
                    "iter": it + 1, "args": vars(args)}, f"{out_dir}/head.pt")
print("done", out_dir)
