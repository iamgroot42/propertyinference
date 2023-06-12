#!/bin/bash

## Run command as:
## sbatch --export=ALL,PROP=1 --job-name=prop1 attacks_synthetic.slurm

# Number of tasks (same as number of scripts)
#SBATCH --ntasks=1

# Allocation group
#SBATCH -A uvasrg_paid

# Total memory requested on system
# Maximum 4 nodes per job allowed
# Maximum 32GB/core allowed
#SBATCH --mem=48G

# Partition of machine
#SBATCH -p gpu

# Which GPU (and how many) to request
# Prefer V100 over 2080
#SBATCH --gres=gpu:v100:1

# Request specific number of CPUs for the task
# Maximum 10 cores per job allowed
#SBATCH --cpus-per-task=9

# Time limit (go for max allowed: 3 days for GPU)
#SBATCH --time=3-00:00:00

# Output file paths (make sure this path exists in the folder where you run the script)
#SBATCH --output=log/maadface/%x-%j.out
#SBATCH --error=log/maadface/%x-%j.err

CUDA_VISIBLE_DEVICES=0 knocky python train_models.py --load_config configs/maadface/train_arcface.json --parallel False --num_models $NMODELS --split $SPLIT --value $VALUE
