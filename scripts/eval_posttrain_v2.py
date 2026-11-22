# Make a long evaluation rollout with the backbone and a head, and save it so that compare_rollout_timing_v2.py
# can compare it with the simulation at full resolution (run on a GPU node from the repo root).
# Usage: PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts python scripts/eval_posttrain_v2.py \
#            --ckpt <S ckpt> --head <head.pt | none> --twall 97 --tag _R3s0
#        PYTHONPATH=... python scripts/compare_rollout_timing_v2.py --ckpt <S ckpt> --twall 97 --neighbors 96 98 --tag _R3s0
# The rollout starts from simulation frames 100-104, like the earlier rollouts, and runs to frame 1999.
# It writes <DATA>/Twall_<T>_rollout<tag>_vapor.npy and _fields128.npy; the comparison script finds them by the tag.
# --head none gives the backbone alone; the pretrained head is <work>/head_pretrained.pt.
import argparse
import numpy as np, torch
from fm_head import Head, Scales
from posttrain_common import DATA, Backbone, load_meta, rollout
from bubbleformer_io import fluid_params, read_frames

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--head", required=True, help="path to a head checkpoint, or none")
p.add_argument("--twall", type=int, default=97)
p.add_argument("--tag", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--frames", type=int, default=1895)
p.add_argument("--data", default=DATA)
args = p.parse_args()

W, START = 5, 100
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed); np.random.seed(args.seed)
head, scales = None, None
if args.head != "none":
    ck = torch.load(args.head, map_location="cpu")
    head = Head(ck["width"]).to(dev).eval(); head.load_state_dict(ck["head"])
    scales = Scales(ck["scales"]).to(dev)
meta = load_meta(args.twall, args.data)
backbone = Backbone(args.ckpt, dev)
x = read_frames(args.data, f"Twall_{args.twall}", START, START + W)[None].to(dev)
ro = rollout(backbone, head, scales, x, fluid_params(meta).to(dev), torch.tensor([float(args.twall)], device=dev),
             meta, -(-args.frames // W), full_vapor=True)
name = f"{args.data}/Twall_{args.twall}_rollout{args.tag}"
np.save(f"{name}_vapor.npy", ro["vapor512"][0, :args.frames])
np.save(f"{name}_fields128.npy", ro["fields128"][0, :args.frames])
print("saved", name, ro["vapor512"].shape)
