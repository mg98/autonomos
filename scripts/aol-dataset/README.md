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
pip install -r requirements.txt
```

Make sure **Java (JDK) 21** is installed on your system.

## Run

You need to create the search index before you can create the dataset.
This will require ~7GB of disk space.

```
python create_index.py
python create_dataset.py
```

Thereafter, you should find a `.csv` file in this folder.
Feel free to `rm -r indexes`.

Finally, t
Move `aol_dataset.csv` inside the [`data`](/data) directory of this repository.

### Instruction for SLURM Users

If you run on SLURM, the corresponding scripts can be invoked by:

```
sbatch ./create_index.sh
# wait for job to finish...
sbatch ./create_dataset.sh
```

For parallel processing of the dataset, consider using:

```
./parallel_create_dataset.sh
```

This will launch 16 instances, each working on different shards of the data, and accordingly generate several CSV files.

After all jobs finished, merge generated CSV files using

```
awk '(NR == 1) || (FNR > 1)' aol_*_*.csv > aol_dataset.csv
rm aol_*_*.csv
```
