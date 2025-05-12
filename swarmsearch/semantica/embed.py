import torch
from transformers import BertTokenizer, BertModel
from allrank.models.model_utils import get_torch_device
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

device = get_torch_device()

model_name = "bert-base-uncased"
max_length = 512
stride = 256

def chunk_tokens(tokens, max_length=512, stride=256):
    """
    Break a token list into overlapping chunks of length up to 'max_length',
    skipping very small leftover pieces unless everything is too small.
    Returns a list of chunks (each a list of token IDs).
    """
    # If tokens fit in one chunk, just return them
    if len(tokens) <= max_length:
        return [tokens]
    
    chunks = []
    seen_valid_chunk = False
    for i in range(0, len(tokens), stride):
        chunk = tokens[i : i + max_length]
        # Only keep reasonably sized chunks
        if len(chunk) > max_length // 4:
            chunks.append(chunk)
            seen_valid_chunk = True

    # Fallback if we didn't produce any chunk
    if not seen_valid_chunk:
        return [tokens]

    return chunks

def embed(text):
    """
    Single-text embedding with chunking.
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    tokens = tokenizer.encode(text, verbose=False)
    # Use our new helper
    token_chunks = chunk_tokens(tokens, max_length=max_length, stride=stride)

    # If there's only one chunk, do old single-chunk logic
    if len(token_chunks) == 1:
        inputs = tokenizer(text, return_tensors="pt", 
                           padding="max_length",
                           truncation=True, 
                           max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    # Otherwise, handle multiple chunks
    chunk_embeddings = []
    for chunk in token_chunks:
        chunk_text = tokenizer.decode(chunk)
        inputs = tokenizer(chunk_text, return_tensors="pt", 
                           padding="max_length", 
                           truncation=True, 
                           max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        chunk_embeddings.append(chunk_embedding)

    # Average across those embeddings
    return sum(chunk_embeddings) / len(chunk_embeddings)


def embed_batch(list_of_texts, batch_size):
    """
    Batch-aware embedding. Splits each text into chunks if necessary,
    collects all chunks, and processes them in batches on the GPU.
    Then averages per-text chunk embeddings to produce one final
    embedding per text.
    """

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Step 1: Convert each text into one or more "chunks" in parallel
    def process_text(idx, text):
        tokens = tokenizer.encode(text, verbose=False)
        token_chunks = chunk_tokens(tokens, max_length=max_length, stride=stride)
        # Track which text (by idx) each chunk corresponds to
        return token_chunks, [idx]*len(token_chunks)
    
    results = Parallel(n_jobs=-1, batch_size=1024)(
        delayed(process_text)(idx, text) 
        for idx, text in tqdm(enumerate(list_of_texts), total=len(list_of_texts), desc="Chunking texts")
    )
    
    # Combine results
    all_chunks = []
    text2chunks = []
    for token_chunks, text_indices in results:
        all_chunks.extend(token_chunks)
        text2chunks.extend(text_indices)

    # Step 2: Embed all chunks in batches
    chunk_embeddings = [None] * len(all_chunks)
    for start_idx in tqdm(range(0, len(all_chunks), batch_size), 
                          total=len(all_chunks) // batch_size, 
                          desc="Embedding chunks"):
        batch_tokens = all_chunks[start_idx : start_idx + batch_size]
        batch_texts = [tokenizer.decode(tokens) for tokens in batch_tokens]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        batch_embs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        for i, emb in enumerate(batch_embs):
            chunk_embeddings[start_idx + i] = emb

    # Step 3: Average chunk embeddings by text
    text_embeddings_map = {}
    for i, emb in enumerate(chunk_embeddings):
        text_idx = text2chunks[i]
        if text_idx not in text_embeddings_map:
            text_embeddings_map[text_idx] = []
        text_embeddings_map[text_idx].append(emb)

    final_embeddings = []
    for idx in range(len(list_of_texts)):
        embeddings_for_text = text_embeddings_map[idx]
        avg_emb = sum(embeddings_for_text) / len(embeddings_for_text)
        final_embeddings.append(avg_emb)

    return np.array(final_embeddings)