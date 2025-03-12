from dart.types import ClickThroughRecord, FeatureVector, TermBasedFeatures
import random
from dataclasses import dataclass
import pandas as pd
from dart.types import Document

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

# def realistic_fake_ctr(df: pd.DataFrame, queries_df: pd.DataFrame, docs_df: pd.DataFrame) -> ClickThroughRecord:
#     """
#     Generates a realistically looking but fake ClickThroughRecord.
#     """
#     # Convert candidate dataframe to list of Document objects
#     candidate_docs = []
#     for _, row in df.sample(10).iterrows():
#         query_index = row['QueryIndex']
#         candi_list = row['CandiList'].split('\t')
        
#         for docid in candi_list:
#             doc_record = docs_df.loc[docid]
#             doc = Document(
#                 id=docid,
#                 title=doc_record['Title'].strip().lower(),
#                 url=doc_record['Url'].strip().lower()
#             )
#             candidate_docs.append(doc)

#     feat = FeatureVector.make()
#     feat.title = rand_term_based_features()
#     feat.url = rand_term_based_features()
#     feat.number_of_slash_in_url = rand_norm_float()
#     feat.pos = rand_norm_float()

#     return ClickThroughRecord(
#         rel=1.0,
#         qid=rand_int(666666, 6666666),
#         feat=feat
#     )

