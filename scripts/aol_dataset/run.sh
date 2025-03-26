#!/bin/bash

sbatch create_index.sh
jobid=$(sbatch create_index.sbatch | awk '{print $4}')
jobid=$(sbatch create_dataset.sbatch --dependency=afterok:"$jobid" | awk '{print $4}')
sbatch postprocess.sbatch --dependency=afterok:"$jobid"
