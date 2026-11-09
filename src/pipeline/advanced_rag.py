import logging
from langchain_core.documents import Document
from src.retrieval.retriever import retrieve
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.pipeline.query_rewriter import rewrite_query
from src.llm.client import call_llm
from config import SYSTEM_PROMPT, TOP_K

logger = logging.getLogger(__name__)

# Initialised once at module load — models are expensive to reload per request
_bm25 = BM25Retriever()
_reranker = CrossEncoderReranker()


def _dense_to_dicts(documents: list[Document], top_k: int) -> list[dict]:
    """
    Converts LangChain Documents to the dict format used by RRF and reranker.
    Adds rank so RRF can treat dense results the same as BM25 results.
    """

    return [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": 0.0,  # cosine score not exposed by LangChain similarity_search
            "rank": rank,
        }
        for rank, doc in enumerate(documents[:top_k])
    ]


def _build_prompt(query: str, chunk: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunk, 1):
        source = chunk["metadata"].get("filename", "unknown")
        block = f"[SOURCE {i} - {source}]\n{chunk['text']}"
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    return f"""Answer the question using ONLY the context provided below.
If the answer is not in the context, say so explicitly.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


def retrieve_stages(query: str, top_k: int = TOP_K) -> dict[str, list[dict]]:
    """
    Stages 2-4 of the pipeline, every stage returned so the retrieval eval can score each one.
    """

    # Stage 2: Parallel retrieval
    dense_docs = retrieve(query, top_k=top_k * 4)
    dense_results = _dense_to_dicts(dense_docs, top_k * 4)
    bm25_results = _bm25.retrieve(query, top_k=top_k * 4)

    # Stage 3: RRF fusion
    fused = reciprocal_rank_fusion(
        ranked_lists=[dense_results, bm25_results], top_k=top_k * 2
    )

    # Stage 4: Rerank
    reranked = _reranker.rerank(query, candidates=fused)

    return {"dense": dense_results, "bm25": bm25_results, "fused": fused, "reranked": reranked}


def run_advanced_rag_pipeline(
    query: str, top_k: int = TOP_K, return_contexts: bool = False
) -> dict:
    """
    Advanced RAG pipeline:
    query → rewrite → dense + BM25 → RRF fusion → rerank → prompt → LLM
    """

    logger.info(f"Advanced RAG pipeline started for : {query[:60]}")

    # Stage 1: Query rewriting
    rewritten = rewrite_query(query)

    # Stages 2-4: dense + BM25 -> RRF fusion -> rerank
    reranked = retrieve_stages(rewritten, top_k)["reranked"]

    if not reranked:
        return {
            "answer": "No relevant documents found in the knoweldge base.",
            "sources": [],
            "chunks_used": 0,
            "query": query,
            "rewritten_query": rewritten,
        }

    # Stage 5: Prompt + LLM
    prompt = _build_prompt(rewritten, reranked)
    answer = call_llm(prompt, system=SYSTEM_PROMPT)

    logger.info("Advanced RAG pipeline complete.")

    result = {
        "answer": answer,
        "sources": [c["metadata"].get("filename", "unknown") for c in reranked],
        "chunks_used": len(reranked),
        "query": query,
        "rewritten_query": rewritten,
    }

    if return_contexts:
        result["contexts"] = [c["text"] for c in reranked]

    return result
