from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
vec = model.encode("hello world")
print(f"Embedding dims: {len(vec)}")   # should print 384

client = chromadb.Client()
col = client.create_collection("test")
print("Chroma OK")

key = os.getenv("OPENROUTER_API_KEY")
print(f"API key loaded: {'yes' if key else 'NO - CHECK .env'}")



# Expected output:

# Embedding dims: 384
# Chroma OK
# API key loaded: yes