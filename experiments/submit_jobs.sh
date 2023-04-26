#!/bin/bash
for i in {2495..4320}
do
   # First do it for 0, so that config data is saved
   # Then, in later ratios, that data will be reused
   sbatch --array=1-10 --export=ALL,PROP=$i,SPLIT=victim,NMODELS=100 --job-name=syn"${i}"v train_models_synthetic.slurm
   sbatch --array=0-10 --export=ALL,PROP=$i,SPLIT=adv,NMODELS=500 --job-name=syn"${i}"a train_models_synthetic.slurm
done
