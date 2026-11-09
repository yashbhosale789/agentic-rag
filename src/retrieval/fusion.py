import logging
from config import RRF_K

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], top_k: int = 50, text_key: str = "text") -> list[dict]:
    """
    Merges N ranked lists into one via Reciprocal Rank Fusion.

    Each document gets score = sum(1 / (RRF_K + rank)) across all lists.
    Documents are deduplicated by text content before scoring.

    Args:
        ranked_lists: List of ranked result lists. Each result must have
                      a 'text' key (or whatever text_key is set to),
                      plus any metadata you want preserved.
        top_k:        How many fused results to return.
        text_key:     Key used to deduplicate documents across lists.

    Returns:
        List of dicts with original fields + 'rrf_score' + 'rrf_rank'.
    """

    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            key = doc[text_key]
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in docs:
                docs[key] = doc

    fused = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:top_k]

    results = []
    for rrf_rank, key in enumerate(fused):
        entry = dict(docs[key])
        entry["rrf_scores"] = scores[key]
        entry["rrf_rank"] = rrf_rank
        results.append(entry)

    logger.info(f"RRD fusion: {len(ranked_list)} lists -> {len(docs)} unique docs -> top {len(results)} returned")

    return results