#!/bin/bash
# Post-training (one GPU): post-train the head with one reward and one seed, then evaluate it.
#   sbatch -J pt-R3s1 scripts/posttrain_run.sbatch R3 1          (reward, seed)
# Training uses the calibrated defaults of scripts/posttrain_v2.py. Run folder: /pub/ynagaraj/bubble_posttrain/runs/<reward>_v2_seed<seed>/.
# Evaluation:
#   - eval_seeds_v2.py: 4 rollouts (seeds 0-3) at Twall 97 and 107, coarse statistics -> results/posttrain/eval_seeds_<reward>s<seed>.md;
#   - seed 0 only: one full-resolution comparison per wall temperature -> results/posttrain/rollout_timing_Twall_<T>_<reward>v2s0.md.
#SBATCH --job-name=pt-run
#SBATCH --account=eecs195221_class_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=results/posttrain/logs/run_%x_%j.out
REWARD=$1; SEED=${2:-0}; NAME=${REWARD}_v2_seed${SEED}
PY=/pub/ynagaraj/envs/bubble/bin/python
CKPT=/pub/ynagaraj/bubbleformer_ckpt/Bubbleformer-S-PB-Subcooled.ckpt
WORK=/pub/ynagaraj/bubble_posttrain
export PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/posttrain_v2.py --ckpt $CKPT --reward $REWARD --seed $SEED --name $NAME || exit 1
cp $WORK/runs/$NAME/log.jsonl results/posttrain/train_log_${REWARD}v2s${SEED}.jsonl
mkdir -p results/posttrain/eval_${REWARD}s${SEED}
$PY scripts/eval_seeds_v2.py --ckpt $CKPT --twalls 97 107 --seeds 0 1 2 3 --heads ${REWARD}=$WORK/runs/$NAME/head.pt \
    --out results/posttrain/eval_${REWARD}s${SEED}
if [ "$SEED" = 0 ]; then
    for T in 97 107; do
        [ $T = 97 ] && NB="96 98" || NB="106 108"
        $PY scripts/eval_posttrain_v2.py --ckpt $CKPT --head $WORK/runs/$NAME/head.pt --twall $T --tag _${REWARD}v2s0
        $PY scripts/compare_rollout_timing_v2.py --ckpt $CKPT --twall $T --neighbors $NB --tag _${REWARD}v2s0 --out results/posttrain
    done
fi
