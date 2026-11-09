import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import logging
from dotenv import load_dotenv
from src.pipeline.advanced_rag import run_advanced_rag_pipeline

load_dotenv()

import os
os.environ["HF_HUB_OFFLINE"] = "1"  # use local cache only, no network calls

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

def main():
    print("\nPersonal Knowledge Base - Advanced RAG Query")
    print("=" * 40)

    query = input("Enter a question: ").strip()
    if not query:
        print("No question entered")
        return

    print("\nThinking...\n")
    result = run_advanced_rag_pipeline(query)

    print("ANSWER")
    print("-" * 40)
    print(result["answer"])
    print()
    print(f"Original Query : {result['query']}")
    print(f"Rewritten Querry : {result['rewritten_query']}")
    print(f"Sources : {'.'.join(set(result['sources']))}")
    print(f"Chunks Used : {result['chunks_used']}")


if __name__ == "__main__":
    main()
