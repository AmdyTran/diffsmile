#!/bin/bash
#SBATCH --job-name=gnot-train
#SBATCH --output=logs/gnot_train_%j.out
#SBATCH --error=logs/gnot_train_%j.err
#SBATCH --partition=gpupr.24h
#SBATCH --account=ls_math
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=nvidia_rtx_pro_6000:1

# Load environment
cd /cluster/project/math/andtran/develop/masters_thesis/code

# Run training with uv
uv run python diffsmile/gnot_lightning.py \
    --embed_dim=256 \
    --lr=0.001 \
    --n_heads=4 \
    --n_layers=6 \
    --n_experts=4 \
    --mlp_layers=4 \
    --batch_size=32 \
    --epochs=500 \
    --calendar_loss_weight=1e-5 \
    --butterfly_loss_weight=1e-5 \
    --smoothness_loss_weight=0.005 \
    --smoothness_weight_mode=inverse_sqrt

