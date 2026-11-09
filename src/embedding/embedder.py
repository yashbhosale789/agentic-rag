from typing import List
from sentence_transformers import SentenceTransformer as SE
from langchain_core.documents import Document
from config import EMBEDDINGS_MODEL_NAME, EMBEDDING_DIAMENSION
import logging
import torch

logger = logging.getLogger(__name__)

# Module-level singleton — load once, reuse across calls.
# SentenceTransformer downloads the model on first run (~90MB for bge-small),
# then caches it in ~/.cache/huggingface/

_model = None

def get_embedding_model() -> SE:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Embedding Model : {EMBEDDINGS_MODEL_NAME} on {device}")
        _model = SE(EMBEDDINGS_MODEL_NAME, device=device)
        logger.info(f"Embedding Model loaded. Output dimension : {EMBEDDING_DIAMENSION}")
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Convert a list of strings to embedding vectors.
    Returns a list of float lists (one vector per text).
    """
    model = get_embedding_model()

    # BGE models need a query prefix at inference time for retrieval tasks.
    # For document embeddings (not queries), no prefix is needed.
    # This is an asymmetric retrieval model — queries and docs are embedded differently.

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,  # normalize to unit length for cosine similarity
        show_progress_bar=len(texts) > 50,  # show progress bar for larger batches
        batch_size=32,
        # device="gpu",  # use GPU if available, else CPU
    )
    return embeddings.tolist()

def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.
    BGE requires the prefix 'Represent this sentence: ' for query-side embeddings.
    Omitting this degrades retrieval quality noticeably.
    """

    model = get_embedding_model()
    prefixed = f"Represent this sentence: {query}"
    embedding = model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=False,
        # device="gpu",  # use GPU if available, else CPU
    )
    return embedding.tolist()
