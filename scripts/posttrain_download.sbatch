#!/bin/bash
# Post-training step 0: download the full-resolution frames of the head-training files (CPU only, one file per task).
#   sbatch scripts/posttrain_download.sbatch          (the 6 default files, as a job array)
# Each task saves <DATA>/Twall_<T>_fields512.npy (about 4.2 GB) and can resume after a failure.
#SBATCH --job-name=pt-download
#SBATCH --account=eecs195221_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=results/posttrain/logs/download_%A_%a.out

TWALLS=(90 94 98 102 106 110)
/pub/ynagaraj/envs/bubble/bin/python scripts/download_fullres_v2.py ${TWALLS[$SLURM_ARRAY_TASK_ID]}
