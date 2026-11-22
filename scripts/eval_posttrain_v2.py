# Make one long evaluation rollout with the backbone and a head, and save it for compare_rollout_timing_v2.py,
# which compares it with the simulation at full resolution (run on a GPU node, from the repo root).
# Usage: PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts python scripts/eval_posttrain_v2.py \
#            --ckpt <S ckpt> --head <head.pt | none> --twall 97 --tag _R3s0
#        PYTHONPATH=... python scripts/compare_rollout_timing_v2.py --ckpt <S ckpt> --twall 97 --neighbors 96 98 --tag _R3s0
# The rollout starts from simulation frames 100-104, like all earlier rollouts, and runs to frame 1999.
# It writes <DATA>/Twall_<T>_rollout<tag>_vapor.npy and _fields128.npy. The comparison script finds them by the tag.
# --head none runs the backbone alone. The pretrained head (before post-training) is <work>/head_pretrained.pt.
import argparse
import numpy as np, torch
from flow_matching_head import ResidualHead, Normalizer
from posttrain_common import DATA, Backbone, load_metadata, run_rollout
from bubbleformer_io import fluid_params, read_frames

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="Bubbleformer-S checkpoint")
parser.add_argument("--head", required=True, help="path to a head checkpoint, or none")
parser.add_argument("--twall", type=int, default=97)
parser.add_argument("--tag", required=True, help="suffix of the output file names, for example _R3s0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--frames", type=int, default=1895)
parser.add_argument("--data", default=DATA)
args = parser.parse_args()

FRAMES, START_FRAME = 5, 100
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)
head, normalizer = None, None
if args.head != "none":
    checkpoint = torch.load(args.head, map_location="cpu")
    head = ResidualHead(checkpoint["width"]).to(device).eval()
    head.load_state_dict(checkpoint["head"])
    normalizer = Normalizer(checkpoint["scales"]).to(device)
metadata = load_metadata(args.twall, args.data)
backbone = Backbone(args.ckpt, device)
start = read_frames(args.data, f"Twall_{args.twall}", START_FRAME, START_FRAME + FRAMES)[None].to(device)
rollout = run_rollout(backbone, head, normalizer, start, fluid_params(metadata).to(device),
                      torch.tensor([float(args.twall)], device=device), metadata, -(-args.frames // FRAMES),
                      save_full_vapor=True)
name = f"{args.data}/Twall_{args.twall}_rollout{args.tag}"
np.save(f"{name}_vapor.npy", rollout["vapor_512"][0, :args.frames])
np.save(f"{name}_fields128.npy", rollout["fields_128"][0, :args.frames])
print("saved", name, rollout["vapor_512"].shape)
