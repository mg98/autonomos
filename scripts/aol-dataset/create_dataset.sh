#!/bin/bash


#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=64

srun zsh -c "echo ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT}"

export JAVA_HOME=~/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH

srun zsh -c "
    ~/miniconda3/envs/pyserini/bin/python ./create_dataset.py \
        --job-id=${SLURM_ARRAY_TASK_ID} \
        --job-count=${SLURM_ARRAY_TASK_COUNT}
"