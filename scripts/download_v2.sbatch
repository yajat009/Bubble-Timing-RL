#!/bin/bash
# Download BubbleML 2.0 files on a CPU node (no GPU needed). Pass the wall temperatures as arguments:
#   sbatch scripts/download_v2.sbatch 107
#SBATCH --job-name=bubble-download
#SBATCH --account=eecs195221_class
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/v2/logs/download_%j.out

/pub/ynagaraj/envs/bubble/bin/python scripts/download_bubbleml_v2.py "$@"
