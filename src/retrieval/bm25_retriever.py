import json
import logging
from pathlib import Path
from rank_bm25 import BM25Okapi
from config import CHUNKS_PATH, TOP_K

logger = logging.getLogger(__name__)

class BM25Retriever:
    """
    Keyword retriever backed by BM25Okapi.

    Loads pre-chunked text from chunks.json produced by ingest.py.
    Tokenisation is whitespace split — sufficient for English prose.
    For production, swap in a proper tokenizer (spaCy, NLTK punkt).
    """

    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self._chunks = self._load_chunks(chunks_path)
        self._corpus = [c["text"] for c in self._chunks]
        self._tokenized = [doc.lower().split() for doc in self._corpus]
        self._index = BM25Okapi(self._tokenized)
        logger.info(f"BM25 index built over {len(self._chunks)} chunks")

    def _load_chunks(self, path: Path) -> list[dict]:
        if not path.exists():
            raise FileNotFoundError(
                f"Chunks.json not found at path {path}. Run scripts/ingest.py first."
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Returns top_k chunks as dicts: {text, metadata, score, rank}.
        Score is a raw BM25 float — only meaningful relative to other scores
        from this same query. Do not compare across queries.
        """
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []

        for rank, idx in enumerate(ranked_indices[:top_k]):
            results.append({
                "text": self._chunks[idx]["text"],
                "metadata": self._chunks[idx]["metadata"],
                "score": float(scores[idx]),
                "rank": rank
            })

        return results
