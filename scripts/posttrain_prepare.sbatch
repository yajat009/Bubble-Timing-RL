#!/bin/bash
# Post-training steps 1-2 (one GPU): prepare the head data, pretrain the head, and evaluate the pretrained head.
#   sbatch scripts/posttrain_prepare.sbatch          (after posttrain_download.sbatch has finished)
# Outputs: /pub/ynagaraj/bubble_posttrain/head_data/, head_pretrained.pt, and the evaluation tables in
# results/posttrain/ (tag _head: backbone + pretrained head, no post-training) for Twall 97 and 107.
#SBATCH --job-name=pt-prepare
#SBATCH --account=eecs195221_class_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=results/posttrain/logs/prepare_%j.out
set -e
PY=/pub/ynagaraj/envs/bubble/bin/python
CKPT=/pub/ynagaraj/bubbleformer_ckpt/Bubbleformer-S-PB-Subcooled.ckpt
WORK=/pub/ynagaraj/bubble_posttrain
export PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/prepare_head_data_v2.py --ckpt $CKPT --twalls 90 94 98 102 106 110
$PY scripts/pretrain_head_v2.py --twalls 90 94 98 102 106 110
set +e
for T in 97 107; do
    [ $T = 97 ] && NB="96 98" || NB="106 108"
    $PY scripts/eval_posttrain_v2.py --ckpt $CKPT --head $WORK/head_pretrained.pt --twall $T --tag _head
    $PY scripts/compare_rollout_timing_v2.py --ckpt $CKPT --twall $T --neighbors $NB --tag _head --out results/posttrain
done
