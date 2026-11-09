import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.retrieval.retriever import retrieve

from src.pipeline.rag import run_rag_pipeline
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_OFFLINE"] = "1"  # use local cache only, no network calls
# print("API KEY LOADED:", os.getenv("OPENROUTER_API_KEY"))

def main():
    print("\nPersonal Knowledge Base — RAG Query")
    print("=" * 40)

    query = input ("Enter a question: ").strip()
    if not query:
        print("No question entered.")
        return

    # Only for testing purpose

    # results = retrieve(query)

    # print(f"\n[Search] Top {len(results)} retrieved chunks:\n")

    # for i, doc in enumerate(results,1):
    #     print(
    #         f"[{i}] Source: {doc.metadata.get('filename', 'unknown')} | "
    #         f"Chunk: {doc.metadata.get('chunk_index', '?')}"
    #     )
    #     print(doc.page_content[:300])
    #     print("-" * 60)

    # Comment out the above block for RAG testing

    print("\nThinking...\n")
    result = run_rag_pipeline(query)

    print("ANSWER")
    print("-" * 40)
    print(result["answer"])
    print()
    print(f"Sources  : {', '.join(set(result['sources']))}")
    print(f"Chunks   : {result['chunks_used']}")

if __name__ == "__main__":
    main()
