import logging
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from config import BASE_DIR, DATA_DIR, CHUNKS_PATH
from src.ingestion.loader import load_directory
from src.ingestion.chunker import chunk_documents
from src.vectorstore.store import add_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def save_chunks(chunks: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "text": chunk.page_content, "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    print(f"Loading documents from : {DATA_DIR}")

    documents = load_directory(DATA_DIR)
    if not documents:
        print("No documents found. Add files to data/documents/ and retry. Exiting.")
        return

    chunks = chunk_documents(documents)

    print(f"\n[INFO] Embedding and storing {len(chunks)} chunks...\n")
    add_chunks(chunks)

    save_chunks(chunks, CHUNKS_PATH)
    print(f"[OK] Chunks saved to : {CHUNKS_PATH}")

    print(f"\n[OK] Ingestion complete.")
    print(f"   Documents loaded : {len(documents)}")
    print(f"   Chunks produced  : {len(chunks)}")
    print(f"   Stored in : storage/chroma/")
    print(f"\nSample chunk (first):")
    print("-" * 60)
    print(chunks[0].page_content[:300])
    print("\nMetadata:", chunks[0].metadata)

    print(f"Created {len(chunks)} chunks from {len(documents)} document(s)")

if __name__ == "__main__":
    main()
