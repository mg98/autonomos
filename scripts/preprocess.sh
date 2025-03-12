#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=64

~/miniconda3/envs/decpy/bin/python ./preprocess.py "$@"
