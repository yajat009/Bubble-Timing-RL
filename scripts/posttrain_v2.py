# Post-train the flow-matching head with a reward (run on a GPU node, from the repo root).
# Usage: PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts python scripts/posttrain_v2.py --ckpt <S ckpt> --reward R3 --seed 0
#
# Rewards (see posttrain_common.REWARD_TERMS):
#   R2 = snapshot statistics: heat-flux histogram and vapor fraction;
#   R3 = timing statistics: departure rate, waiting times between departures, and lengths of quiet periods;
#   R4 = R2 and R3 together.
#
# Each iteration:
#   1. Pick one training file and one start state (5 true frames at 512 x 512, from Twall_<T>_fields512.npy).
#   2. Generate a group of rollouts from that state with the sampling copy of the head.
#   3. Score each rollout against the simulation at the same wall temperature. Each distance is divided by its
#      noise scale (posttrain_common.reference_statistics), and the reward is minus their sum.
#   4. Turn the rewards into weights between 0 and 1: a rollout above the group mean gets a weight above 0.5,
#      a rollout below the mean gets a weight below 0.5.
#   5. Update the head with the DiffusionNFT loss on the kept steps of each rollout (noise, sample, condition):
#        positive velocity = (1 - beta) * sampling head + beta * head
#        negative velocity = (1 + beta) * sampling head - beta * head
#        loss = (1 / beta) * mean[weight * |positive - target|^2 + (1 - weight) * |negative - target|^2]
#      where target = sample - noise. This pulls the head toward samples of high-reward rollouts and pushes it
#      away from samples of low-reward rollouts. Two more terms keep it close to the data and to its start:
#        + lambda_sup * flow-matching loss on true training pairs
#        + lambda_ref * |head - pretrained head|^2
#   6. Move the sampling copy a small step toward the trained head (exponential moving average).
# The update follows ReAl and DiffusionNFT, with the proposal's settings (beta 0.1, lambda_sup 0.1, lambda_ref 0.01).
import argparse, copy, json, os, time
import numpy as np, torch
from flow_matching_head import ResidualHead, Normalizer, flow_matching_loss
from posttrain_common import (DATA, Backbone, load_metadata, run_rollout, rollout_statistics,
                              distances_to_reference, compute_reward, reference_statistics, REWARD_TERMS)
from bubbleformer_io import fluid_params

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="Bubbleformer-S checkpoint")
parser.add_argument("--reward", choices=list(REWARD_TERMS), required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--twalls", type=int, nargs="+", default=None, help="training files; default: those of the pretrained head")
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--group", type=int, default=8, help="rollouts per group")
parser.add_argument("--bundles", type=int, default=120, help="steps of 5 frames per rollout (120 steps = 600 frames)")
parser.add_argument("--keep", type=int, default=24, help="steps per rollout kept for the update")
parser.add_argument("--epochs", type=int, default=2, help="passes over the kept steps per iteration")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--lambda-sup", type=float, default=0.1)
parser.add_argument("--lambda-ref", type=float, default=0.01)
parser.add_argument("--ema", type=float, default=0.9, help="sampling head <- ema * sampling head + (1 - ema) * head")
parser.add_argument("--data", default=DATA)
parser.add_argument("--work", default="/pub/ynagaraj/bubble_posttrain")
args = parser.parse_args()

FRAMES, START_FRAME = 5, 100
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)

# Three copies of the head: the one that is trained, the sampling copy that generates rollouts,
# and the frozen pretrained head for the lambda_ref term.
pretrained = torch.load(f"{args.work}/head_pretrained.pt", map_location="cpu")
normalizer = Normalizer(pretrained["scales"]).to(device)
train_walls = args.twalls or pretrained["twalls"]
head = ResidualHead(pretrained["width"]).to(device)
head.load_state_dict(pretrained["head"])
head_pretrained = copy.deepcopy(head).eval().requires_grad_(False)
head_sampling = copy.deepcopy(head).eval().requires_grad_(False)
optimizer = torch.optim.AdamW(head.parameters(), args.lr, weight_decay=0.0)
backbone = Backbone(args.ckpt, device)
rollout_frames = FRAMES * args.bundles

print("reference statistics ...", flush=True)
references = {t: reference_statistics(t, rollout_frames, args.data) for t in train_walls}
for t, ref in references.items():
    print(f"Twall {t}: {ref['statistics']['departure_rate']:.2f} departures per 100 frames; noise scales "
          + ", ".join(f"{k} {v:.3g}" for k, v in ref["noise_scales"].items()), flush=True)

# True training pairs for the lambda_sup term (the same data as the pretraining).
supervised_files = [(t, np.load(f"{args.work}/head_data/Twall_{t}_pred128.npy", mmap_mode="r"),
                     np.load(f"{args.data}/Twall_{t}_fields128.npy", mmap_mode="r")) for t in train_walls]


def supervised_batch(size):
    """Normalized true residuals and their conditions, for `size` random windows of one random training file."""
    wall_temp, predictions, fields = supervised_files[np.random.randint(len(supervised_files))]
    window = np.random.randint(0, len(predictions), size)
    first = START_FRAME + FRAMES * window                         # first input frame of each window
    prediction = torch.from_numpy(predictions[window].astype(np.float32)).to(device)
    truth = torch.from_numpy(np.stack([fields[f + FRAMES:f + 2 * FRAMES] for f in first]).astype(np.float32)).to(device)
    last_input = torch.from_numpy(np.stack([fields[f + FRAMES - 1] for f in first]).astype(np.float32)).to(device)
    condition = normalizer.build_condition(prediction, last_input, torch.full((size,), float(wall_temp), device=device))
    return normalizer.to_normalized_residual(truth - prediction), condition


start_frames = {t: np.load(f"{args.data}/Twall_{t}_fields512.npy", mmap_mode="r") for t in train_walls}
run_dir = f"{args.work}/runs/{args.reward}_seed{args.seed}"
os.makedirs(run_dir, exist_ok=True)
log_file = open(f"{run_dir}/log.jsonl", "a")
t0 = time.time()
for iteration in range(args.iters):
    # 1-2. One start state, a group of rollouts from it with the sampling head.
    wall_temp = int(np.random.choice(train_walls))
    metadata = load_metadata(wall_temp, args.data)
    start = int(np.random.randint(START_FRAME, len(start_frames[wall_temp]) - rollout_frames - FRAMES))
    frames = torch.from_numpy(start_frames[wall_temp][start:start + FRAMES].astype(np.float32)).to(device)
    frames = frames[None].expand(args.group, -1, -1, -1, -1).contiguous()
    fluid = fluid_params(metadata).to(device).expand(args.group, -1)
    walls = torch.full((args.group,), float(wall_temp), device=device)
    try:
        rollouts = run_rollout(backbone, head_sampling, normalizer, frames, fluid, walls, metadata, args.bundles,
                               bundles_to_keep=args.keep)
    except FloatingPointError as error:
        print(f"iteration {iteration}: {error}; skipped", flush=True)
        continue

    # 3. Score each rollout.
    reference = references[wall_temp]
    statistics = [rollout_statistics(rollouts["heat_flux"][g], rollouts["vapor_128"][g], reference["quiet_threshold"])
                  for g in range(args.group)]
    distances = [distances_to_reference(s, reference["statistics"]) for s in statistics]
    rewards = np.array([compute_reward(args.reward, d, reference["noise_scales"]) for d in distances])

    # 4. Weights from the rewards, relative to the group.
    advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    weights = np.clip(0.5 + 0.25 * advantage, 0.0, 1.0)

    # 5. DiffusionNFT update. kept_steps holds one (noise, sample, condition) triple per kept step, each for the whole
    # group, so the weights repeat once per kept step.
    noise = torch.cat([step[0] for step in rollouts["kept_steps"]])
    samples = torch.cat([step[1] for step in rollouts["kept_steps"]])
    conditions = torch.cat([step[2] for step in rollouts["kept_steps"]])
    sample_weights = torch.from_numpy(np.tile(weights, len(rollouts["kept_steps"]))).float()
    head.train()
    losses = []
    for _ in range(args.epochs):
        for batch in torch.randperm(len(noise)).split(args.batch):
            z0, z1 = noise[batch].to(device).float(), samples[batch].to(device).float()
            condition, weight = conditions[batch].to(device).float(), sample_weights[batch].to(device)
            tau = torch.rand(len(batch), device=device)
            z_tau = (1 - tau.view(-1, 1, 1, 1)) * z0 + tau.view(-1, 1, 1, 1) * z1
            target = z1 - z0
            velocity = head(z_tau, tau, condition)
            with torch.no_grad():
                velocity_sampling = head_sampling(z_tau, tau, condition)
                velocity_pretrained = head_pretrained(z_tau, tau, condition)
            positive = (1 - args.beta) * velocity_sampling + args.beta * velocity
            negative = (1 + args.beta) * velocity_sampling - args.beta * velocity
            loss_positive = (positive - target).pow(2).mean(dim=(1, 2, 3))
            loss_negative = (negative - target).pow(2).mean(dim=(1, 2, 3))
            loss_align = (weight * loss_positive + (1 - weight) * loss_negative).mean() / args.beta
            loss = (loss_align
                    + args.lambda_sup * flow_matching_loss(head, *supervised_batch(args.batch))
                    + args.lambda_ref * (velocity - velocity_pretrained).pow(2).mean())
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
            losses.append(loss.item())

    # 6. The sampling head follows the trained head.
    with torch.no_grad():
        for p_sampling, p_trained in zip(head_sampling.parameters(), head.parameters()):
            p_sampling.mul_(args.ema).add_((1 - args.ema) * p_trained)

    row = {"iteration": iteration, "twall": wall_temp, "start_frame": start,
           "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()), "loss": float(np.mean(losses)),
           "departure_rate_rollouts": float(np.mean([s["departure_rate"] for s in statistics])),
           "departure_rate_simulation": reference["statistics"]["departure_rate"], "time_s": round(time.time() - t0)}
    for term in REWARD_TERMS["R4"]:
        row[f"distance_{term}"] = float(np.mean([d[term] for d in distances]))
    log_file.write(json.dumps(row) + "\n"); log_file.flush()
    print(json.dumps(row), flush=True)
    if (iteration + 1) % 10 == 0 or iteration + 1 == args.iters:
        torch.save({"head": head.state_dict(), "width": pretrained["width"], "scales": pretrained["scales"],
                    "twalls": train_walls, "iteration": iteration + 1, "args": vars(args)}, f"{run_dir}/head.pt")
print("done", run_dir)
