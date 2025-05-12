#!/bin/bash

#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=64

module load cuda12.3/toolkit/12.3

rm -rf data/doc_embeddings.lmdb
srun zsh -c "
    ~/miniconda3/envs/decpy/bin/python get_docs.py \
        --job-id=${SLURM_ARRAY_TASK_ID} \
        --job-count=${SLURM_ARRAY_TASK_COUNT}
"