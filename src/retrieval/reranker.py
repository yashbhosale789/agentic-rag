import logging
from sentence_transformers import CrossEncoder
from config import TOP_K, RERANKER_MODEL

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Precision stage: re-scores RRF candidates with a cross-encoder.

    Takes top-50 from RRF, returns top_k with accurate relevance scores.
    Model is loaded once at init — do not instantiate per request.
    """

    def __init__(self, model_name: str = RERANKER_MODEL, top_k: int = TOP_K):
        self._model = CrossEncoder(model_name)
        self._top_k = top_k
        logger.info(f"CrossEncoder Reranker loaded: {model_name}")


    def rerank(self, query: str, candidates: list[dict], text_key: str = "text") -> list[dict]:
        """
        Args:
            query:      The search query (rewritten or raw).
            candidates: RRF-fused results. Each must have a text_key field.
            text_key:   Key containing chunk text.

        Returns:
            top_k results sorted by cross-encoder score descending.
            Each result gets 'ce_score' and 'ce_rank' added.
        """

        if not candidates:
            return[]

        pairs = [(query, doc[text_key]) for doc in candidates]
        scores = self._model.predict(pairs)

        scored = [
            {**doc, "ce_score": float(score)}
            for doc, score in zip(candidates, scores)
        ]

        ranked = sorted(scored, key=lambda x: x["ce_score"], reverse=True)

        for rank, doc in enumerate(ranked):
            doc["ce_rank"] = rank

        result = ranked[:self._top_k]

        logger.info(
            "Reranker: %d candidates → top %d returned (best score: %.4f)",
            len(candidates),
            len(result),
            result[0]["ce_score"] if result else 0.0,
        )
        return result
