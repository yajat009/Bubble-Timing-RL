#!/bin/bash
# Run GPU tests A-D on HPC3 in one job. Submit from the repo root:  sbatch scripts/run_gpu_tests_v2.sbatch
#   A. accuracy_check_v2.py   Is the Bubbleformer setup correct? (short-term error against "repeat the last frame")
#   D. rollout_starts_v2.py   Does the rollout lose bubble activity from 4 other start frames?
#   B. dwell_times_v2.py      How long do the simulation and the rollouts stay quiet or active? (uses A's and D's rollouts)
#   C. window_sweep_v2.py     Does a window longer than 5 frames help predict the next departure?
# A failing test does not stop the others. Each test writes its table to results/v2 and its log to results/v2/logs.
#SBATCH --job-name=bubble-tests
#SBATCH --account=eecs195221_class_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=results/v2/logs/run_gpu_tests_%j.out

PY=/pub/ynagaraj/envs/bubble/bin/python
CKPT=/pub/ynagaraj/bubbleformer_ckpt/Bubbleformer-S-PB-Subcooled.ckpt
export PYTHONPATH=/pub/ynagaraj/Bubbleformer:scripts
mkdir -p results/v2/logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run() {   # run <name> <command...>: time one test and keep going if it fails
    local name=$1; shift
    echo "=== $name: start $(date +%T)"
    "$@" > "results/v2/logs/$name.log" 2>&1
    echo "=== $name: exit code $?, end $(date +%T)"
}

run A_accuracy_check $PY scripts/accuracy_check_v2.py --ckpt $CKPT --twall 97
run D_rollout_starts $PY scripts/rollout_starts_v2.py --ckpt $CKPT --twall 97 --starts 250 500 750 995 --frames 1000
run B_dwell_times    $PY scripts/dwell_times_v2.py --twall 97 --rollouts Twall_97_rollout \
                         Twall_97_rollout_start250 Twall_97_rollout_start500 Twall_97_rollout_start750 Twall_97_rollout_start995
run C_window_sweep   $PY scripts/window_sweep_v2.py --windows 1 2 5 10 20 --seeds 3
