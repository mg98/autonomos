#!/bin/bash

for ((start=0; start<=1500000; start+=100000)); do
    end=$((start + 100000))
    echo "Processing chunk: $start to $end"
    sbatch scripts/create_dataset.sh --start "$start" --end "$end"
done