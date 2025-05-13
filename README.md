# 🕸️🔍 SwarmSearch

This repository collects the code used to generate the dataset and experimental results for _"SwarmSearch: Decentralized Search Engine with Self-Funding Economy"_ [[preprint](https://arxiv.org/pdf/2505.07452)].

## 💾 Dataset

Please refer to [`./scripts/aol_dataset`](./scripts/aol_dataset) for detailed instructions on how to compile the AOL dataset used throughout this project.

## 📦 Installation

This project requires **Python 3.9**.
Please run the following command to install dependencies.

```bash
make install
```

## ⚙️ Preprocessing

In addition to the base dataset, for reasons of efficiency, we also precompile DART feature vectors or _clickthrough records (CTR)_. This step is only needed for _Experiment #2_, and can be skipped otherwise.

To this end, please run

```bash
python scripts/get_ctrs.py
```

This takes a long time. We recommend running it parallelized in a SLURM environment, for which we provide `scripts/get_ctrs.sbatch`.

Each instance `i` will generate `data/ctrs_{i}.lmdb`.
When finished, please run `python scripts/combine_ctrs.py` to combine into a single `data/ctrs.lmdb`.

## 🔬 Experiment #1: Retrieval Accuracy

tba

## 🔬 Experiment #2: Spam Prevention

In order to reproduce the results described in Section VII.B, please run the following command:

```
python scripts/exp_spam_prevention.py
```

For parallelized execution, use our SLURM script `sbatch scripts/exp_spam_prevention.sbatch`.

Results generate very slowly due to the approximation of Shapley values.
Our scripts (both `.py` and `.sbatch`) are designed in a way that their execution can be interrupted and re-continued at any time without corruption of the result.
The script can be run until the results achieve the desired resolution, as exhausting it over the entire dataset can be intractable.

## 🔬 Experiment #3: Resilience Against Sybil Attacks

tba

## 📈 Plotting

We used R for the visualization of our results, see [`./plotting.R`](./plotting.R).
