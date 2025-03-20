#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=64

export JAVA_HOME=~/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH

~/miniconda3/envs/pyserini/bin/python ./create_index.py
