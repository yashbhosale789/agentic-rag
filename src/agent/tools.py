import json
import logging
from pathlib import Path
from typing import Any
from config import CHUNKS_PATH
from src.pipeline.advanced_rag import retrieve_stages
from src.pipeline.query_rewriter import rewrite_query

logger = logging.getLogger(__name__)


def _format_chunks_for_agent(chunks: list[str]) -> str:
    if not chunks:
        return "No relevant information found in the document."

    parts = []
    for i, text in enumerate(chunks, start=1):
        parts.append(f"[CHUNK {i}]\n{text.strip()}")

    return "\n\n".join(parts)


def build_retriever_tool() -> dict[str, Any]:

    def retrieve(query: str) -> str:
        logger.info(f"Tool called — query: '{query}'")
        try:
            # rewrite + hybrid retrieval + rerank; the pipeline's answer step is skipped (the agent answers)
            chunks = [c["text"] for c in retrieve_stages(rewrite_query(query))["reranked"]]
            logger.info(f"Retrieved {len(chunks)} chunks")
            return _format_chunks_for_agent(chunks)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return f"Retrieval error: {str(e)}"

    return {
        "name": "retrieve",
        "description": (
            "Search the knowledge base (documents of five companies: ZX Bank, Aventro Motors, "
            "Cendara University, Velvera Technologies and CloudWay-24) for relevant information. "
            "Input must be a concise search query of 5-10 words. "
            "Use this tool whenever the question requires specific facts from the document."
        ),
        "function": retrieve,
    }


def _load_chunks() -> list[dict[str, Any]]:
    path = Path(CHUNKS_PATH)
    if not path.exists():
        logger.warning(f"Chunks file not found at {CHUNKS_PATH}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_metadata_tool() -> dict[str, Any]:

    def get_document_info(_: str) -> str:
        logger.info("Metadata tool called")

        try:
            chunks = _load_chunks()
            if not chunks:
                return "No document metadata available."

            filenames = {
                Path(c.get("metadata", {}).get("filename", "unknown")).name for c in chunks
            }

            # counts only: listing all 236 file names would cost ~2.5K tokens per call
            return (
                f"Knowledge base contents:\n"
                f"- Source documents: {len(filenames)}\n"
                f"- Total chunks indexed: {len(chunks)}\n"
                f"- Domain: documents of five fictional companies: "
                f"ZX Bank, Aventro Motors, Cendara University, Velvera Technologies, CloudWay-24."
            )

        except Exception as e:
            logger.error(f"Metadata tool failed: {e}")
            return f"Metadata error: {str(e)}"

    return {
        "name": "get_document_info",
        "description": (
            "Returns metadata about the knowledge base - the source document name, "
            "total chunk count and a summary of document contents. "
            "Use this tool when the question is about what document exists in the "
            "knowledge base, not about specific content within it."
        ),
        "function": get_document_info,
    }


def get_all_tools() -> list[dict[str, Any]]:
    return [build_retriever_tool(), build_metadata_tool()]
