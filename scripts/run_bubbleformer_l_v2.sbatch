#!/bin/bash
# Does Bubbleformer-L also lose bubble activity in long rollouts? The paper reports that L is much better than S
# on subcooled boiling, so the stall found with S (T6, tests A-D) must be checked with L before the RL work.
# Submit from the repo root, after Twall_107 is downloaded (scripts/download_bubbleml_v2.py 107):
#   sbatch scripts/run_bubbleformer_l_v2.sbatch          (all steps)
#   sbatch scripts/run_bubbleformer_l_v2.sbatch 97       (steps 1-2 only; no Twall 107 data needed)
#   sbatch scripts/run_bubbleformer_l_v2.sbatch 107      (steps 3-4 only)
#   1. L short-term accuracy at Twall 97 (5 start frames).
#   2. L rollout at Twall 97, compared with the simulation and Twall 96 and 98.
#   3. L rollout at Twall 107 (also not used in training), compared with Twall 106 and 108.
#   4. S rollout at Twall 107, so that S and L can be compared on the second held-out file.
# Outputs use the tags _L and _S in their names, so the earlier S results at Twall 97 stay as they are.
#SBATCH --job-name=bubble-L
#SBATCH --account=eecs195221_class_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=results/v2/logs/run_bubbleformer_l_%j.out

PY=/pub/ynagaraj/envs/bubble/bin/python
CKPT_L=/pub/ynagaraj/bubbleformer_ckpt/Bubbleformer-L-PB-Subcooled.ckpt
CKPT_S=/pub/ynagaraj/bubbleformer_ckpt/Bubbleformer-S-PB-Subcooled.ckpt
export PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts
mkdir -p results/v2/logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run() {   # run <name> <command...>: time one step and keep going if it fails
    local name=$1; shift
    echo "=== $name: start $(date +%T)"
    "$@" > "results/v2/logs/$name.log" 2>&1
    echo "=== $name: exit code $?, end $(date +%T)"
}

PART=${1:-all}
if [ "$PART" != 107 ]; then
    run L_accuracy_97 $PY scripts/accuracy_check_v2.py --ckpt $CKPT_L --twall 97 --starts 5 --tag _L
    run L_rollout_97  $PY scripts/compare_rollout_timing_v2.py --ckpt $CKPT_L --twall 97 --neighbors 96 98 --tag _L
fi
if [ "$PART" != 97 ]; then
    run L_rollout_107 $PY scripts/compare_rollout_timing_v2.py --ckpt $CKPT_L --twall 107 --neighbors 106 108 --tag _L
    run S_rollout_107 $PY scripts/compare_rollout_timing_v2.py --ckpt $CKPT_S --twall 107 --neighbors 106 108 --tag _S
fi
