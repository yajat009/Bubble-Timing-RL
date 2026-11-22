#!/bin/bash
# Post-training step 3 (one GPU): post-train the head with one reward, then evaluate it on Twall 97 and 107.
#   sbatch scripts/posttrain_run.sbatch R2 0          (reward, seed)
#   sbatch scripts/posttrain_run.sbatch R3 0
# Training log and head: /pub/ynagaraj/bubble_posttrain/runs/<reward>_seed<seed>/; a copy of the log goes to
# results/posttrain/. Evaluation tables: results/posttrain/rollout_timing_Twall_<T>_<reward>s<seed>.md
#SBATCH --job-name=pt-run
#SBATCH --account=eecs195221_class_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/posttrain/logs/run_%x_%j.out
REWARD=$1; SEED=${2:-0}; TAG=_${REWARD}s${SEED}
PY=/pub/ynagaraj/envs/bubble/bin/python
CKPT=/pub/ynagaraj/bubbleformer_ckpt/Bubbleformer-S-PB-Subcooled.ckpt
WORK=/pub/ynagaraj/bubble_posttrain
export PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY scripts/posttrain_v2.py --ckpt $CKPT --reward $REWARD --seed $SEED || exit 1
cp $WORK/runs/${REWARD}_seed${SEED}/log.jsonl results/posttrain/train_log${TAG}.jsonl
for T in 97 107; do
    [ $T = 97 ] && NB="96 98" || NB="106 108"
    $PY scripts/eval_posttrain_v2.py --ckpt $CKPT --head $WORK/runs/${REWARD}_seed${SEED}/head.pt --twall $T --tag $TAG --seed $SEED
    $PY scripts/compare_rollout_timing_v2.py --ckpt $CKPT --twall $T --neighbors $NB --tag $TAG --out results/posttrain
done
