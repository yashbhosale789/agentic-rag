from typing import List
from langchain_core.documents import Document
from src.retrieval.retriever import retrieve
from src.llm.client import call_llm
from config import SYSTEM_PROMPT
import logging

logger = logging.getLogger(__name__)

# SYSTEM_PROMPT = """You are a precise question-answering assistant.
# You answer questions strictly based on the provided context.
# If the context does not contain enough information to answer the question, 
# say "I don't have enough information in my knowledge base to answer this."
# Do not use any knowledge outside the provided context."""


def build_promt(query: str, chunks: List[Document]) -> str:
    """
    Construct the RAG prompt: labeled context blocks + user query.

    Source labeling serves two purposes:
    1. Helps the model distinguish between sources
    2. Enables citation in the answer ("according to SOURCE 1...")
    """

    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("filename", "unknown")
        block = f"[SOURCE {i} - {source}]\n{chunk.page_content}"
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = f"""Answer the question using ONLY the context provided below.
If the answer is not in the context, say so explicitly.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    return prompt

def run_rag_pipeline(query: str) -> dict:
    """
    Full RAG pipeline: query → retrieve → prompt → LLM → answer.

    Returns a dict with the answer and the source chunks for transparency.
    """

    logger.info(f"Running RAG Pipeline for : '{query[:60]}'")

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query)
    if not chunks:
        return {
            "answer":"No relevant documents found in the knoweldge base",
            "sources":[],
            "query": query
        }

    # Step 2: Build grounded prompt
    prompt = build_promt(query, chunks)

    answer = call_llm(prompt, system=SYSTEM_PROMPT)

    logger.info("RAG pipeline complete.")

    return {
        "answer": answer,
        "sources": [c.metadata.get("filename","unknown") for c in chunks],
        "chunks_used": len(chunks),
        "query": query
    }