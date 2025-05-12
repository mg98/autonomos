# Generating AOL Dataset with Ranked Results

These scripts use [`pyserini`](https://github.com/castorini/pyserini) to compile a top-10 list of search results for each query in the AOL search logs.
The selection is based on BM25 while ensuring that the _clicked_ document is included in the list.

This setup assumes the AOL-IOA dataset is completely downloaded within the `ir_datasets` framework (located in `~/.ir_datasets/aol-ia`).
If this is not the case, please visit follow the instructions in the [aolia-tools](https://github.com/terrierteam/aolia-tools) repository.

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

The creation of the dataset is composed of three steps:

1. Indexing
2. Dataset creation
3. Post-processing

If you run on SLURM, the corresponding scripts can be invoked with:

```
make 
```
