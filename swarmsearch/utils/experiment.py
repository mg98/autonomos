from argparse import ArgumentParser
from typing import Callable
from swarmsearch.datasets.aol import load_dataset
from swarmsearch.utils.cache import Cache
from swarmsearch.utils.data import compile_clickthrough_records
from allrank.config import Config
from swarmsearch.dart.utils import compute_feature_stats
import pandas as pd
import torch
import numpy as np
from joblib import Parallel, delayed

# Set random seeds for reproducibility and job id consistency
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if hasattr(torch, 'mps'):
    torch.mps.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Experiment:
    def __init__(self, id: str, fn: Callable[["Experiment", str], list], user_ids_source: str = None):
        """
        Initialize the experiment.
        
        Args:
            id: ID of the experiment (used for file naming)
            fn: Function that generates results given a user ID
        """
        parser = ArgumentParser(description=f"User simulation of experiment {id}")
        parser.add_argument('--user', '-u', type=int, help='User ID to simulate (if not specified, all users will be simulated)')
        parser.add_argument('--job-id', type=int, help='Slurm job ID')
        parser.add_argument('--job-count', type=int, help='Total number of jobs')
        
        self.args = parser.parse_args()
        self.config = Config.from_json("./allRank_config.json")
        self._setup_cache()
        self.export_path = f"results/experiment_{id}.tsv"
        self.fn = fn
        self.user_ids_source = user_ids_source

    @property
    def user_ids(self) -> set[str]:
        """
        Returns user IDs to simulate in this experiment run.
        """

        if self.args.user is not None:
            return {self.args.user}
        
        user_ids = set()
        if self.user_ids_source is not None:
            shapley_df = pd.read_csv(self.user_ids_source, sep='\t', header=None)
            user_ids = set(shapley_df[0])
        else:
            user_ids = set(self.cache.get("user_ids"))

        if self.args.job_id is not None:
            user_ids = set(user_id for user_id in user_ids if user_id % self.args.job_count == self.args.job_id)

        return user_ids - self.get_done_user_ids()

    def _setup_cache(self):
        self.cache = Cache()
        if self.cache.is_empty():
            print("Initializing cache...")
            df = load_dataset()
            sample_df = df.sample(n=1000, random_state=42)
            sample_ctrs = compile_clickthrough_records(sample_df)
            feature_means, feature_stds = compute_feature_stats(sample_ctrs)
            self.cache.set("user_ids", df["user_id"].unique())
            self.cache.set("feature_means", feature_means)
            self.cache.set("feature_stds", feature_stds)
            print("Cache set")
    
    def get_done_user_ids(self) -> set[str]:
        try:
            results_df = pd.read_csv(self.export_path, sep='\t', header=None)
            return set(results_df[0])
        except FileNotFoundError:
            return set()

    def run(self, parallel: bool = False):
        def process_user(user_id):
            results = self.fn(self, user_id)

            sep = "\t"
            result_string = sep.join([str(r) for r in results])
            if self.args.job_id is not None:
                with open(self.export_path, "a") as f:
                    f.write(result_string + "\n")
            else:
                print(result_string)

        Parallel(n_jobs=8 if parallel else 1)(
                delayed(process_user)(user_id) for user_id in self.user_ids
            )