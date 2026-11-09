from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into retrieval-sized chunks.

    Returns a new list of Documents where each item is a chunk,
    carrying forward all metadata from the parent document plus
    a chunk index for ordering/debugging.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,                    # character count (not token count)
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index per source document for traceability
    source_counters: dict = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        source_counters[source] = source_counters.get(source, 0) + 1
        chunk.metadata["chunk_index"] = source_counters[source]

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} document(s)")
    return chunks
