#!/bin/bash
#SBATCH --job-name=minitaur_demo
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=minitaur_demo_%j.out

# Paths on JASMIN (group workspace, not home)
WORK=/gws/ssde/j25b/iecdt/cwarg
export HF_HOME=$WORK/hf_cache
export HF_HUB_OFFLINE=1   # model is pre-downloaded; compute nodes need no internet

source ~/miniforge3/etc/profile.d/conda.sh
conda activate $WORK/envs/minitaur

nvidia-smi
python $WORK/thessaly/minitaur_demo.py
