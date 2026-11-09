import logging
from src.llm.client import call_llm
from config import REWRITE_SYSTEM

logger = logging.getLogger(__name__)

def rewrite_query(query: str) -> str:
    """
    Rewrites a conversational query into a precise search query.

    Falls back to original query if rewriting fails — retrieval still works,
    just with lower quality input.

    Args:
        query: Raw user query.

    Returns:
        Rewritten query string.
    """

    try:
        rewritten = call_llm(prompt=query, system=REWRITE_SYSTEM)
        logger.info(f"Query rewrite: {query} -> {rewritten}")
        return rewritten

    except Exception as e:
        logger.info(f"Query rewrite failed, using original. Rason: {e}")
        return query