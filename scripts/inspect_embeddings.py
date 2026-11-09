import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from config import STORAGE_DIR, CHROMA_COLLECTION

client = chromadb.PersistentClient(path=str(STORAGE_DIR))
collection = client.get_collection(CHROMA_COLLECTION)

results = collection.get(include=["embeddings", "documents", "metadatas"])

print(f"Total chunks stored: {len(results['ids'])}\n")

for i, (id_, doc, meta, emb) in enumerate(
    zip(
        results["ids"],
        results["documents"],
        results["metadatas"],
        results["embeddings"],
    )
):
    print(f"--- Chunk {i+1} ---")
    print(f"ID       : {id_}")
    print(f"Source   : {meta.get('filename')}, chunk {meta.get('chunk_index')}")
    print(f"Text     : {doc[:150]}...")
    print(f"\n Embedding: dim={len(emb)}")
    row_size = 8
    for row_start in range(0, len(emb), row_size):
        row = emb[row_start : row_start + row_size]
        print(f"  [{row_start:>3}]  " + "  ".join(f"{v:>10.6f}" for v in row))
    print(f"\n Vector   : {[round(v, 6) for v in emb]}")
    print()
