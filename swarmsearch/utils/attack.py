from swarmsearch.dart.types import ClickThroughRecord, FeatureVector, TermBasedFeatures
import random
from dataclasses import dataclass
import pandas as pd
from typing import Callable

seed = 0

def rand_norm_float():
    global seed
    rng = random.Random(seed)
    seed += 1
    return rng.random()

def rand_int(min, max):
    global seed
    rng = random.Random(seed)
    seed += 1
    return rng.randint(min, max)

@dataclass
class Statistics:
    min: float
    max: float
    mean: float
    variance: float
    sum: float

    @property
    def features(self):
        return [self.min, self.max, self.mean, self.variance, self.sum]

def rand_statistics():
    return Statistics(
        min=rand_norm_float(),
        max=rand_norm_float(),
        mean=rand_norm_float(),
        variance=rand_norm_float(),
        sum=rand_norm_float()
    )

def rand_term_based_features():
    return TermBasedFeatures(
        bm25=rand_norm_float(),
        tf=rand_statistics(),
        idf=rand_statistics(),
        tf_idf=rand_statistics(),
        stream_length=rand_statistics(),
        stream_length_normalized_tf=rand_statistics(),
        cos_sim=rand_norm_float(),
        covered_query_term_number=rand_norm_float(),
        covered_query_term_ratio=rand_norm_float(),
        char_len=rand_norm_float(),
        term_len=rand_norm_float(),
        total_query_terms=rand_norm_float(),
        exact_match=rand_norm_float(),
        match_ratio=rand_norm_float()
    )

def rand_ctr() -> ClickThroughRecord:
    """
    Generate a ClickThroughRecord with random feature values.
    """
    feat = FeatureVector()
    feat.title = rand_term_based_features()
    feat.url = rand_term_based_features()
    feat.number_of_slash_in_url = rand_norm_float()

    return ClickThroughRecord(
        rel=rand_norm_float() > 0.5,
        qid=rand_int(666666, 6666666),
        feat=feat
    )

def flip_label(ctr: ClickThroughRecord) -> ClickThroughRecord:
    return ClickThroughRecord(rel=not ctr.rel, qid=ctr.qid, feat=ctr.feat)

def poison_ctrs(
        poison_fn: Callable[[ClickThroughRecord], ClickThroughRecord],
        ctrs: list[ClickThroughRecord], 
        ratio: float = 1.0
        ) -> list[ClickThroughRecord]:
    poison_count = int(len(ctrs) * ratio)
    keep_count = len(ctrs) - poison_count
    poison_ctrs = list(map(poison_fn, ctrs[keep_count:]))
    return ctrs[:keep_count] + poison_ctrs
