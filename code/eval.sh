#!/bin/bash
#SBATCH --job-name=gnot-train
#SBATCH --output=logs/gnot_train_%j.out
#SBATCH --error=logs/gnot_train_%j.err
#SBATCH --partition=gpupr.24h
#SBATCH --account=ls_math
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=nvidia_rtx_pro_6000:1

# Load environment
cd /cluster/project/math/andtran/develop/masters_thesis/code

# Run training with uv
uv run python run_eval_aggregate_job.py 
