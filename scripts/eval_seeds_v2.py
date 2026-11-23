# How much do the evaluation numbers change from one random rollout to the next? (run on a GPU node, from the repo root)
# Usage: PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts python scripts/eval_seeds_v2.py \
#            --ckpt <S ckpt> --heads head=<path> R2=<path> R3=<path> --twalls 97 107 --seeds 0 1 2 3
#
# The head is random, so two evaluation rollouts of the same head differ. Before two heads can be compared, we need
# this spread. For each head, wall temperature and seed, the script makes one 1,895-frame rollout from frames 100-104,
# computes the coarse statistics that the rewards use (posttrain_common.py), and reports the mean and the standard
# deviation over the seeds. The coarse detector agrees with the full-resolution detector within about 4%.
# Output: <OUT>/eval_seeds.md and eval_seeds.json.
import argparse, json, os
import numpy as np, torch
from flow_matching_head import ResidualHead, Normalizer
from posttrain_common import (DATA, Backbone, load_metadata, run_rollout, rollout_statistics,
                              distances_to_reference, reference_statistics)
from bubbleformer_io import fluid_params, read_frames

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="Bubbleformer-S checkpoint")
parser.add_argument("--heads", nargs="+", required=True, help="name=path pairs; path none = backbone alone")
parser.add_argument("--twalls", type=int, nargs="+", default=[97, 107])
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--frames", type=int, default=1895)
parser.add_argument("--data", default=DATA)
parser.add_argument("--out", default="results/posttrain")
args = parser.parse_args()

FRAMES, START_FRAME = 5, 100
device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = Backbone(args.ckpt, device)
results = {}
for wall_temp in args.twalls:
    metadata = load_metadata(wall_temp, args.data)
    reference = reference_statistics(wall_temp, args.frames, args.data, n_samples=1)
    start = read_frames(args.data, f"Twall_{wall_temp}", START_FRAME, START_FRAME + FRAMES)[None].to(device)
    for spec in args.heads:
        name, path = spec.split("=", 1)
        head, normalizer = None, None
        if path != "none":
            checkpoint = torch.load(path, map_location="cpu")
            head = ResidualHead(checkpoint["width"]).to(device).eval()
            head.load_state_dict(checkpoint["head"])
            normalizer = Normalizer(checkpoint["scales"]).to(device)
        rows = []
        for seed in (args.seeds if head is not None else args.seeds[:1]):   # the backbone alone is not random
            torch.manual_seed(seed); np.random.seed(seed)
            rollout = run_rollout(backbone, head, normalizer, start, fluid_params(metadata).to(device),
                                  torch.tensor([float(wall_temp)], device=device), metadata, -(-args.frames // FRAMES))
            statistics = rollout_statistics(rollout["heat_flux"][0, :args.frames], rollout["vapor_128"][0, :args.frames],
                                            reference["quiet_threshold"])
            row = distances_to_reference(statistics, reference["statistics"])
            row["departure_rate"] = statistics["departure_rate"]
            rows.append(row)
            print(f"Twall {wall_temp}, {name}, seed {seed}: " + ", ".join(f"{k} {v:.3g}" for k, v in row.items()), flush=True)
        results[f"{wall_temp}/{name}"] = rows
    results[f"{wall_temp}/simulation_rate"] = reference["statistics"]["departure_rate"]

columns = ["departure_rate", "waiting_times", "quiet_periods", "heat_flux_histogram", "vapor_fraction"]
lines = ["# Evaluation spread over seeds (coarse 128 x 128 statistics)", "",
         "Mean +- standard deviation over seeds of one 1,895-frame rollout each. departure_rate is per 100 frames; "
         "the other columns are distances to the simulation (0 = same).", ""]
for wall_temp in args.twalls:
    lines += [f"## Twall {wall_temp} (simulation: {results[f'{wall_temp}/simulation_rate']:.2f} departures per 100 frames)", "",
              "| Head | Seeds | " + " | ".join(columns) + " |", "|---" * (len(columns) + 2) + "|"]
    for spec in args.heads:
        name = spec.split("=", 1)[0]
        rows = results[f"{wall_temp}/{name}"]
        cells = [f"{np.mean([r[c] for r in rows]):.3g} +- {np.std([r[c] for r in rows]):.2g}" for c in columns]
        lines.append(f"| {name} | {len(rows)} | " + " | ".join(cells) + " |")
    lines.append("")
table = "\n".join(lines)
print(table)
os.makedirs(args.out, exist_ok=True)
open(f"{args.out}/eval_seeds.md", "w").write(table + "\n")
json.dump(results, open(f"{args.out}/eval_seeds.json", "w"), indent=1)
