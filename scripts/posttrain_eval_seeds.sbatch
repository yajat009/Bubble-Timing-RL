#!/bin/bash
# Measure how much the evaluation numbers change between random rollouts of the same head (one GPU).
#   sbatch scripts/posttrain_eval_seeds.sbatch
# Heads: backbone alone, pretrained head, R2 seed 0, R3 seed 0; Twall 97 and 107; seeds 0-3.
#SBATCH --job-name=pt-evalseeds
#SBATCH --account=eecs195221_class_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/posttrain/logs/eval_seeds_%j.out
PY=/pub/ynagaraj/envs/bubble/bin/python
CKPT=/pub/ynagaraj/bubbleformer_ckpt/Bubbleformer-S-PB-Subcooled.ckpt
WORK=/pub/ynagaraj/bubble_posttrain
export PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts
$PY scripts/eval_seeds_v2.py --ckpt $CKPT --twalls 97 107 --seeds 0 1 2 3 \
    --heads S=none head=$WORK/head_pretrained.pt R2=$WORK/runs/R2_seed0/head.pt R3=$WORK/runs/R3_seed0/head.pt
