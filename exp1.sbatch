#!/bin/bash

#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0

module load cuda12.3/toolkit/12.3

srun zsh -c "
    ~/miniconda3/envs/decpy/bin/python exp1.py \
        --job-id=${SLURM_ARRAY_TASK_ID} \
        --job-count=${SLURM_ARRAY_TASK_COUNT}
"
