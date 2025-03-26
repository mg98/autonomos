#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0
#SBATCH --cpus-per-task=64

export JAVA_HOME=~/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH

conda init bash
source ~/.bashrc
conda activate pyserini

awk '(NR == 1) || (FNR > 1)' aol_*_*.csv > aol_dataset.csv
rm aol_*_*.csv
python postprocess.py
rm aol_dataset.csv
