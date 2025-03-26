# Generating AOL Dataset with Ranked Results

These scripts use [`pyserini`](https://github.com/castorini/pyserini) to compile a top-10 list of search results for each query in the AOL search logs.
The selection is based on BM25 while ensuring that the _clicked_ document is included in the list.

> [!NOTE]  
> This setup requires Python 3.10. We recommend creating a dedicated environment for its execution as it will not be compatible with the environment created for running the main scripts of this repository.

- Requires Python 3.10 and JDK 21

## Installation

Install project dependencies in an environment with **Python 3.10**.

```
conda create -n pyserini python=3.10
conda activate pyserini
make install
```

Make sure **Java (JDK) 21** is installed on your system.

### Instructions for SLURM Users

The creation of the dataset is composed of four steps:
1. Indexing
2. Parallel proce
If you run on SLURM, the corresponding scripts can be invoked with:

```
make 
```

The last command will launch 16 instances, each working on different shards of the data, and accordingly generate several CSV files.

After all jobs finished, merge the generated CSV files using

```
awk '(NR == 1) || (FNR > 1)' aol_*_*.csv > aol_dataset.csv
rm aol_*_*.csv
```
