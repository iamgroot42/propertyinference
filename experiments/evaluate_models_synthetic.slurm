#!/bin/bash

# Number of tasks (will be 1 unless you split job and want parallel execution)
#SBATCH --ntasks=1

# Allocation group
#SBATCH -A uvasrg

# Total memory requested on system
# Maximum 4 nodes per job allowed
# Maximum 32GB/core allowed
#SBATCH --mem=24G

# Partition of machine
#SBATCH -p gpu

# Which GPU (and how many) to request
# Prefer V100 over 2080
#SBATCH --gres=gpu:a100:1

# Request specific number of CPUs for the task
# Maximum 10 cores per job allowed
#SBATCH --cpus-per-task=4

# Time limit (go for max allowed: 3 days for GPU)
#SBATCH --time=3-00:00:00

# Output file paths
#SBATCH --output=log/exps_synthetic/%x-%j.out
#SBATCH --error=log/exps_synthetic/%x-%j.err

# Run all ratios at once
CUDA_VISIBLE_DEVICES=0 knocky python blackbox_attacks.py --load_config configs/synthetic/bb_attacks.json --en "bb_$SLURM_ARRAY_TASK_ID" --prop $SLURM_ARRAY_TASK_ID
