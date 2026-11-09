from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)

from config import DATA_GLOB

import logging

logger = logging.getLogger(__name__)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": UnstructuredWordDocumentLoader
}

def load_documents(file_path: Path) -> List[Document]:
    """
    Load a single file into a list of Documents.
    
    Each loader may return multiple documents - PDF's return one per page,
    text files typically return one for the whole file.
    """

    suffix = file_path.suffix.lower()
    loader_cls = LOADER_MAP.get(suffix)

    if not loader_cls:
        logger.warning(f"Unsupported file type : {file_path.name}. Skipping file.")
        return []

    try:
        loader = loader_cls(str(file_path))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["filename"] = str(file_path)

        logger.info(f"Loaded {len(docs)} document(s) from {file_path.name}")
        return docs

    except Exception as e:
        logger.error(f"Failed to load {file_path.name} : {e}")
        return []

def load_directory(directory: Path) -> List[Document]:
    """
    Recursively load all supported files from a directory.
    """
    all_docs: List[Document] = []

    files = [f for f in directory.rglob(DATA_GLOB) if f.is_file() and f.suffix.lower() in LOADER_MAP]

    if not files:
        logger.warning(f"No supported files found in directory: {directory}")
        return []

    for file_path in files:
        docs = load_documents(file_path)
        all_docs.extend(docs)

    logger.info(f"Total documents loaded: {len(all_docs)} from {len(files)} file(s)")
    return all_docs
