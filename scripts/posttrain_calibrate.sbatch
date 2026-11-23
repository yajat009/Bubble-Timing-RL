#!/bin/bash
# Check that a stronger update makes the training reward rise (one GPU), before the full set of runs.
#   sbatch scripts/posttrain_calibrate.sbatch R3
# Compared with the first runs: a 6 times higher learning rate, beta 0.5 instead of 0.1, 4 passes over 32 kept
# steps instead of 2 over 24, a faster-following sampling head (ema 0.5), a smaller supervised term (0.02), and
# 12 fixed start states that repeat every 12 iterations, so the reward of the same start can be compared over time.
#SBATCH --job-name=pt-calibrate
#SBATCH --account=eecs195221_class_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/posttrain/logs/calibrate_%j.out
REWARD=${1:-R3}
PY=/pub/ynagaraj/envs/bubble/bin/python
CKPT=/pub/ynagaraj/bubbleformer_ckpt/Bubbleformer-S-PB-Subcooled.ckpt
WORK=/pub/ynagaraj/bubble_posttrain
export PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts
$PY scripts/posttrain_v2.py --ckpt $CKPT --reward $REWARD --seed 0 --name ${REWARD}_calibrate \
    --lr 3e-4 --beta 0.5 --epochs 4 --keep 32 --ema 0.5 --lambda-sup 0.02 --fixed-starts 12 --iters 48 --bundles 100
cp $WORK/runs/${REWARD}_calibrate/log.jsonl results/posttrain/train_log_${REWARD}_calibrate.jsonl
