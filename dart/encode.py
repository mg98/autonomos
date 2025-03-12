# import re
# from dart.types import FeatureVector, Corpus, TermBasedFeatures
# from ..datasets.aol4ps import Document

# def encode_features(query: str, candidates: list[Document], chosen_index: int) -> list[FeatureVector]:
#     title_corpus = Corpus({doc.id: doc.title for doc in candidates})
#     url_corpus = Corpus({doc.id: doc.url for doc in candidates})
    
#     for doc in candidates:
#         v = FeatureVector()
#         v.title = TermBasedFeatures.make(title_corpus, query, doc.id)
#         v.url = TermBasedFeatures.make(url_corpus, query, doc.id)
#         yield v