from typing import List
from langchain_core.documents import Document
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from config import STORAGE_DIR, CHROMA_COLLECTION, EMBEDDINGS_MODEL_NAME, EMBEDDING_DIAMENSION
import logging

logger = logging.getLogger(__name__)

class BGEEmbeddingAdapter(Embeddings):
    """
    LangChain requires an Embeddings interface object to talk to Chroma.
    We wrap our local BGE functions to satisfy that interface.

    This adapter pattern lets us swap embedding models without touching
    the vector store code — just swap the adapter.
    """

    def embed_documents(self, texts:List[str]) -> List[List[float]]:
        from src.embedding.embedder import embed_texts
        return embed_texts(texts)

    def embed_query(self,text:str) -> List[float]:
        from src.embedding.embedder import embed_query
        return embed_query(text)

def get_vectorstore() -> Chroma:
    """
    Return a Chroma instance backed by local disk storage.
    If the collection already exists, loads it. If not, creates it.
    This is idempotent — safe to call multiple times.
    """

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=BGEEmbeddingAdapter(),
        persist_directory=str(STORAGE_DIR),
    )

def add_chunks(chunks: List[Document]) -> None:
    """
    Embed and store a list of chunks into Chroma.
    Chroma auto-persists to disk — no explicit save() call needed.
    """

    if not chunks:
        logger.warning("No chunks to add to vector store. Skipping.")
        return

    vectorstore = get_vectorstore()

    logger.info(f"Embedding and storing {len(chunks)} chunks to vector store : {CHROMA_COLLECTION}")
    vectorstore.add_documents(documents=chunks)
    logger.info(f"Stored {len(chunks)} chunks to vector store : {CHROMA_COLLECTION}")