import re
import random
import numpy as np
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
import hashlib
from ir_datasets.datasets.aol_ia import AolIaDoc

def tokenize(text):
    return re.findall(r'\b\w+\b', text)

class TFIDF:
    def __init__(self, corpus: Dict[str, str]):
        """
        Initialize a TF-IDF model with a corpus of documents.
        The model computes TF-IDF vectors for all documents in the corpus and
        provides methods to calculate term weights and document similarity.
        
        Args:
            corpus: A dictionary mapping document IDs to document text.
                   Each document will be vectorized and indexed for retrieval.
        """
        self.corpus = {doc_id: text for doc_id, text in corpus.items()}
        if all(not doc.strip() for doc in self.corpus.values()):
            return
        
        self.vectorizer = TfidfVectorizer(
            use_idf=True, smooth_idf=True, norm=None,
            token_pattern=r'(?u)\b\w\w*\b|[0-9]+'
        )
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus.values())
        except Exception as e:
            print(self.corpus.values())
            raise e
        self.feature_names = self.vectorizer.get_feature_names_out()
        # Add these hash maps for O(1) lookups
        self.feature_to_idx = {term: idx for idx, term in enumerate(self.feature_names)}
        self.doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.corpus.keys())}
        # Precompute term counts for each document
        self.term_counts = [doc.split() for doc in self.corpus.values()]
        self.total_terms = [len(doc) for doc in self.term_counts]
        # Add vector cache
        self._vector_cache = {}


    def get_tf_idf(self, doc_id: str, term: str) -> dict[str, float]:
        if all(not doc.strip() for doc in self.corpus.values()):
            return {"tf": 0, "tf_idf": 0, "idf": 0}
        
        try:
            word_idx = self.feature_to_idx[term]
        except KeyError:
            return {"tf": 0, "tf_idf": 0, "idf": 0}
        
        doc_idx = self.doc_to_idx[doc_id]
        tf_idf = self.tfidf_matrix[doc_idx, word_idx]
        idf = self.vectorizer.idf_[word_idx]
        tf = tf_idf / idf if idf != 0 else 0
        
        return {"tf": tf, "tf_idf": tf_idf, "idf": idf}
    
    def get_vector(self, query: str) -> np.ndarray:
        if query not in self._vector_cache:
            self._vector_cache[query] = self.vectorizer.transform([query]).toarray()[0]
        return self._vector_cache[query]

    def get_cos_sim(self, doc_id: str, query_terms: list[str]) -> float:
        """Compute cosine similarity between the query and a document."""
        if doc_id not in self.doc_to_idx:
            return 0.0
        
        query = ' '.join(query_terms)
        query_vector = self.get_vector(query)
        doc_idx = self.doc_to_idx[doc_id]
        document_vector = self.tfidf_matrix[doc_idx]
        dot_product = document_vector.dot(query_vector)[0]
        query_magnitude = np.linalg.norm(query_vector)
        document_magnitude = np.linalg.norm(document_vector.toarray())
        
        if query_magnitude == 0 or document_magnitude == 0:
            return 0.0
        return dot_product / (query_magnitude * document_magnitude)
    
    def get_document_text(self, doc_id: str) -> str:
        """Get the original document text for a given document ID."""
        return self.corpus[doc_id]

class Corpus:
    def __init__(self, docs: Dict[str, str]):
        self.docs = docs
        self.tfidf = TFIDF(docs)
        if not self.is_empty():
            self.bm25 = BM25Okapi([tokenize(t) for t in docs.values()])

    def is_empty(self) -> bool:
        return len(self.docs) == 0 or all(not doc.strip() for doc in self.docs.values())


class Statistics:
    min: float = 0.0
    max: float = 0.0
    sum: float = 0.0
    mean: float = 0.0
    variance: float = 0.0

    @classmethod
    def make(cls, values: list[float] = []):
        v = cls()
        if len(values) == 0:
            return v
        v.min = min(values)
        v.max = max(values)
        v.sum = sum(values)
        v.mean = v.sum / len(values)
        v.variance = sum((x - v.mean) ** 2 for x in values) / len(values)
        return v
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        v = cls()
        v.min = float(arr[0])
        v.max = float(arr[1])
        v.sum = float(arr[2])
        v.mean = float(arr[3])
        v.variance = float(arr[4])
        return v
    
    @property
    def features(self) -> list[float]:
        return [
            self.min,
            self.max,
            self.mean,
            self.variance,
            self.sum
        ]

@dataclass
class TermBasedFeatures:
    bm25: float = 0.0

    tf: Statistics = Statistics()
    idf: Statistics = Statistics()
    tf_idf: Statistics = Statistics()
    stream_length: Statistics = Statistics()
    stream_length_normalized_tf: Statistics = Statistics()

    cos_sim: float = 0.0
    covered_query_term_number: int = 0
    covered_query_term_ratio: float = 0.0
    char_len: int = 0
    term_len: int = 0
    total_query_terms: int = 0
    exact_match: int = 0
    match_ratio: float = 0.0

    @classmethod
    def make(cls, corpus: Corpus, query: str, doc_id: str) -> 'TermBasedFeatures':
        v = cls()
        if corpus.is_empty():
            return v

        query_terms = tokenize(query)
        doc_text = corpus.docs[doc_id]
        doc_text_terms = tokenize(doc_text)

        doc_index = list(corpus.docs.keys()).index(doc_id)
        v.bm25 = corpus.bm25.get_batch_scores(query_terms, [doc_index])[0]

        tfidf_results = [corpus.tfidf.get_tf_idf(doc_id, term) for term in query_terms]

        v.tf = Statistics.make([r["tf"] for r in tfidf_results])
        v.idf = Statistics.make([r["idf"] for r in tfidf_results])
        v.tf_idf = Statistics.make([r["tf_idf"] for r in tfidf_results])
        
        v.stream_length = Statistics.make([len(term) for term in doc_text_terms])
        if len(doc_text_terms) > 0:
            v.stream_length_normalized_tf = Statistics.make([sum(r["tf"] for r in tfidf_results) / len(doc_text_terms)])
        else:
            v.stream_length_normalized_tf = Statistics()
        
        v.cos_sim = corpus.tfidf.get_cos_sim(doc_id, query_terms)

        v.covered_query_term_number = sum(1 for r in tfidf_results if r["tf"] > 0)
        v.covered_query_term_ratio = v.covered_query_term_number / len(query_terms)

        # Get document text from tfidf to calculate lengths
        v.char_len = len(doc_text)
        v.term_len = len(tokenize(doc_text))
        
        # Boolean features
        document_terms = tokenize(doc_text)
        matched_terms = set(query_terms) & set(document_terms)
        match_count = len(matched_terms)
        v.total_query_terms = len(query_terms)
        v.exact_match = 1 if match_count == v.total_query_terms else 0
        v.match_ratio = match_count / v.total_query_terms if v.total_query_terms > 0 else 0

        return v
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        v = cls()
        v.bm25 = float(arr[0])
        v.tf = Statistics.from_array(arr[1:6])
        v.idf = Statistics.from_array(arr[6:11])
        v.tf_idf = Statistics.from_array(arr[11:16])
        return v
    
    @property
    def features(self) -> list[float]:
        return [
            self.bm25, # 0
            *self.tf.features, # 1-5
            *self.idf.features, # 6-10
            *self.tf_idf.features, # 11-15
            *self.stream_length.features, # 16-20
            *self.stream_length_normalized_tf.features, # 21-25
            self.cos_sim, # 26
            self.covered_query_term_number, # 27
            self.covered_query_term_ratio, # 28
            self.char_len, # 29
            self.term_len, # 30
            self.total_query_terms, # 31
            self.exact_match, # 32
            self.match_ratio # 33
        ]


class FeatureVector:
    title: TermBasedFeatures = TermBasedFeatures() # 0-33
    body: TermBasedFeatures = TermBasedFeatures() # 34-67
    url: TermBasedFeatures = TermBasedFeatures() # 68-101
    number_of_slash_in_url: int = 0 # 102

    @classmethod
    def make(cls, candidate_docs: list[AolIaDoc], doc: AolIaDoc, query: str) -> 'FeatureVector':
        v = cls()
        title_corpus = Corpus({ doc.doc_id: doc.title.lower().strip() for doc in candidate_docs })
        body_corpus = Corpus({ doc.doc_id: doc.text.lower().strip() for doc in candidate_docs })
        url_corpus = Corpus({ doc.doc_id: doc.url.lower().strip() for doc in candidate_docs })
        v.title = TermBasedFeatures.make(title_corpus, query, doc.doc_id)
        v.body = TermBasedFeatures.make(body_corpus, query, doc.doc_id)
        v.url = TermBasedFeatures.make(url_corpus, query, doc.doc_id)
        v.number_of_slash_in_url = url_corpus.docs[doc.doc_id].count('/')
        return v
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        v = cls()
        v.title = TermBasedFeatures.from_array(arr[:34])
        v.body = TermBasedFeatures.from_array(arr[34:68])
        v.url = TermBasedFeatures.from_array(arr[68:102])
        v.number_of_slash_in_url = int(arr[102])
        return v

    @property
    def features(self):
        return [
            *self.title.features,
            *self.body.features,
            *self.url.features,
            self.number_of_slash_in_url
        ]
    
    @classmethod
    def n_features(cls) -> int:
        return len(cls().features)

    def __str__(self):
        return ' '.join(f'{i}:{val}' for i, val in enumerate(self.features))


def bounded_string_to_int_hash(x, N=1000000) -> int:
    return int(hashlib.sha256(str(x).encode()).hexdigest(), 16) % N

class ClickThroughRecord:
    rel: bool
    qid: int
    feat: FeatureVector

    def __init__(self, rel=False, qid=0, feat=None):
        self.rel = rel
        self.qid = qid
        self.feat = feat
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        return cls(
            rel=bool(arr[0]),
            qid=int(arr[1]),
            feat=FeatureVector.from_array(arr[2:])
        )

    def to_dict(self):
        return {
            'rel': self.rel,
            'qid': self.qid,
            'feat': self.feat
        }
    
    def to_array(self) -> np.ndarray:
        return np.array([self.rel, self.qid, *self.feat.features], dtype=np.float32)
    

    def __str__(self):
        return f'{int(self.rel)} qid:{self.qid} {self.feat}'
    
@dataclass
class SplitDataset:
    train: list[ClickThroughRecord]
    vali: list[ClickThroughRecord]
    test: list[ClickThroughRecord]

    def _shuffle_maintain_groups(self, records: list[ClickThroughRecord]) -> list[ClickThroughRecord]:
        """
        Helper method to shuffle records while maintaining query groups.
        
        Args:
            records: List of ClickThroughRecord objects
            
        Returns:
            Shuffled list with query groups preserved
        """
        # Group records by query ID
        query_groups = {}
        for record in records:
            if record.qid not in query_groups:
                query_groups[record.qid] = []
            query_groups[record.qid].append(record)
        
        # Get list of query IDs and shuffle their order
        query_ids = list(query_groups.keys())
        random.shuffle(query_ids)
        
        # Optionally shuffle records within each query group
        for qid in query_ids:
            random.shuffle(query_groups[qid])
        
        # Reconstruct the list with shuffled query order
        shuffled_records = []
        for qid in query_ids:
            shuffled_records.extend(query_groups[qid])
            
        return shuffled_records

    def shuffle(self):
        self.train = self._shuffle_maintain_groups(self.train)
        self.vali = self._shuffle_maintain_groups(self.vali)
        self.test = self._shuffle_maintain_groups(self.test)

    def to_dict(self):
        return {
            'train': self.train,
            'vali': self.vali,
            'test': self.test
        }

@dataclass
class Dataset:
    context: list[ClickThroughRecord]
    test: list[ClickThroughRecord]

    def split(self) -> SplitDataset:        
        context_copy = self.context.copy()
        random.shuffle(context_copy)
        total_context = len(context_copy)
        train_size = int((8/9) * total_context)
        train_records = context_copy[:train_size]
        vali_records = context_copy[train_size:]
        return SplitDataset(train_records, vali_records, self.test)