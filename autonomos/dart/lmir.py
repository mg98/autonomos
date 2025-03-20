import numpy as np
from autonomos.dart.utils import tokenize

def collection_frequency(tokenized_docs: list[str], query_term: str) -> int:
    """
    How many times `query_term` occurs in the entire corpus.
    """
    return sum(tk_doc.count(query_term) for tk_doc in tokenized_docs)

def lmir_abs(docs: list[str], doc_terms: list[str], query_terms: list[str]) -> float:
    """
    Language Model for IR with Absolute Discounting Smoothing (LMIR.ABS)
    """
    delta = 0.7
    tokenized_docs = list(map(tokenize, docs))
    total_corpus_stream_length = sum(len(tk_doc) for tk_doc in tokenized_docs)
    
    # P(q_i | D)
    probabilities = [
        max(doc_terms.count(q) - delta, 0) / len(doc_terms) + 
        delta * collection_frequency(tokenized_docs, q) / total_corpus_stream_length
        for q in query_terms
    ]
    
    return float(np.sum(np.log(probabilities)))

def lmir_dir(docs: list[str], doc_terms: list[str], query_terms: list[str]) -> float:
    """
    Language Model for IR with Dirichlet Smoothing (LMIR.DIR)
    """
    mu = 500
    tokenized_docs = list(map(tokenize, docs))
    total_corpus_stream_length = sum(len(tk_doc) for tk_doc in tokenized_docs)

    # P(q_i | D)
    probabilities = [
        (doc_terms.count(q) + mu * (collection_frequency(tokenized_docs, q) / total_corpus_stream_length)) / (len(doc_terms) + mu)
        for q in query_terms
    ]
    
    return float(np.sum(np.log(probabilities)))

def lmir_jm(docs: list[str], doc_terms: list[str], query_terms: list[str]) -> float:
    """
    Language Model for IR with Jelinek-Mercer Smoothing (LMIR.JM)
    """
    lambda_ = 0.1
    tokenized_docs = list(map(tokenize, docs))
    total_corpus_stream_length = sum(len(tk_doc) for tk_doc in tokenized_docs)

    # P(q_i | D)
    probabilities = [
        (1 - lambda_) * doc_terms.count(q) / len(doc_terms) +
        lambda_ * collection_frequency(tokenized_docs, q) / total_corpus_stream_length
        for q in query_terms
    ]
    
    return float(np.sum(np.log(probabilities)))
    
    
    