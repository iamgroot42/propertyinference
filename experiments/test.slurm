#!/bin/bash
#SBATCH --job-name="CelebA Training"
#SBATCH --ntasks=1
#SBATCH -A uvasrg_paid

# Total memory requested on system
# Maximum 4 nodes per job allowed
# Maximum 32GB/core allowed
#SBATCH --mem-per-cpu=8G

# Partition of machine
#SBATCH -p gpu

# Which GPU (and how many) to request
# Prefer V100 over 2080
#SBATCH --gres=gpu:v100:1

# Request specific number of CPUs for the task
# Maximum 10 cores per job allowed
#SBATCH --cpus-per-task=10

# Time limit (go for max allowed: 3 days for GPU)
#SBATCH --time=2-00:00:00

# Output file paths
#SBATCH --output="r%a.out"

OFFSETS=(0 50 100 150 200 250 300 350 400 450)
#RATIOS=("0.0 0.1 0.2 0.3 0.4 0.5","0.6 0.7 0.8 0.9 1.0")
# Your command goes here
echo $SLURM_ARRAY_TASK_ID

#sleep $((${OFFSETS[$(($SLURM_ARRAY_TASK_ID%10))]}/10))
./do_all.sh "0.7 0.8" ${OFFSETS[$SLURM_ARRAY_TASK_ID]} 
#python train_more.py --load_config "./cel_mouth.json" --ratios ${RATIOS[$(($SLURM_ARRAY_TASK_ID/10))]} --split adv --offset ${OFFSETS[$(($SLURM_ARRAY_TASK_ID%10))]} 
wait
echo "finished"