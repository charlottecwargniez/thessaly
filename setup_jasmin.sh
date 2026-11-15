#!/bin/bash
# One-off setup on JASMIN for minitaur_demo. Run on a sci server (e.g. sci-ph-02), NOT on your Mac
# and NOT inside a Slurm job (compute nodes have no internet).
# Puts the conda env and the Hugging Face cache in the group workspace, since home is near its 100G quota.
set -euo pipefail

WORK=/gws/ssde/j25b/iecdt/cwarg
export HF_HOME=$WORK/hf_cache

source ~/miniforge3/etc/profile.d/conda.sh

# 1. Python env (skipped if it already exists)
if [ ! -d "$WORK/envs/minitaur" ]; then
    conda create -y -p "$WORK/envs/minitaur" python=3.11
fi
conda activate "$WORK/envs/minitaur"
pip install --upgrade torch transformers accelerate huggingface_hub

# 2. Download Minitaur-8B (~16 GB of BF16 weights) into the group workspace cache
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("marcelbinz/Llama-3.1-Minitaur-8B")
print("Model downloaded to", path)
PY

# 3. Copy the scripts next to the env
mkdir -p "$WORK/thessaly"
echo "Setup done. Copy minitaur_demo.py and minitaur_demo.sbatch to $WORK/thessaly, then: sbatch minitaur_demo.sbatch"
