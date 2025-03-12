#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=64

module load cuda12.3/toolkit/12.3

PYTHONPATH=/home/mgregori/aol4ps-ltr-dataset \
~/miniconda3/envs/decpy/bin/python preprocess_docs.py
