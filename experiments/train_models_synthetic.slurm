#!/bin/bash

# Number of tasks (will be 1 unless you split job and want parallel execution)
#SBATCH --ntasks=1

# Allocation group
#SBATCH -A uvasrg_paid

# Total memory requested on system
# Maximum 4 nodes per job allowed
# Maximum 32GB/core allowed
#SBATCH --mem=4G

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
#SBATCH --output=log/training_synthetic/%x-%j.out
#SBATCH --error=log/training_synthetic/%x-%j.err

RATIOS=(
    0.0
    0.1
    0.2
    0.3
    0.4
    0.5
    0.6
    0.7
    0.8
    0.9
    1.0
)

# Run all ratios at once
CUDA_VISIBLE_DEVICES=0 knocky python train_models.py --load_config configs/synthetic/train_models.json --split $SPLIT --num_models $NMODELS --prop $PROP --value "${RATIOS[SLURM_ARRAY_TASK_ID]}"
