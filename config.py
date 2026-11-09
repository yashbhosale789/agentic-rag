import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
# RAG_DATA_DIR: read a corpus in place (e.g. the 236-PDF RAG-Multi-Corpus folder)
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data" / "documents"))
DATA_GLOB = os.getenv("RAG_DATA_GLOB", "*")  # e.g. "*.pdf": skip a corpus folder's LICENSE/SOURCE.md
STORAGE_DIR = BASE_DIR / "storage" / "chroma"

# Embeddings
EMBEDDINGS_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIAMENSION = 384

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHUNKS_PATH = BASE_DIR / "storage" / "chunks.json"

# Retrieval
TOP_K = 5
RRF_K = 60
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chroma
CHROMA_COLLECTION = "personal_kb"

# -----------------------------------------------------------------------
# Pipeline LLM — switch PIPELINE_PROVIDER to change model
# -----------------------------------------------------------------------
PIPELINE_PROVIDER = "groq"  # "openrouter" | "groq" | "gemini"

# OpenRouter — 50 req/day free
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
# Other free options:
# "inclusionai/ling-3.0-flash:free"
# "nvidia/nemotron-3-nano-30b-a3b:free"
# "poolside/laguna-s-2.1:free"

# Groq — 14,400 req/day free, fastest inference
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = "GROQ_API_KEY"
GROQ_PIPELINE_MODEL = "llama-3.3-70b-versatile"
# Other free options:
# "llama-3.1-8b-instant"     — fastest, weaker
# "openai/gpt-oss-20b"       — fast, OpenAI open-weight
# "openai/gpt-oss-120b"      — strongest on Groq

# Gemini — 1,500 req/day free via AI Studio
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
GEMINI_API_KEY = "GEMINI_API_KEY"
GEMINI_PIPELINE_MODEL = "gemini-2.5-flash"
# Other free options:
# "gemini-2.5-flash-lite"    — faster, cheaper, slightly weaker

# -----------------------------------------------------------------------
# Evaluator LLM — switch EVAL_PROVIDER to change evaluator
# -----------------------------------------------------------------------
EVAL_PROVIDER = "groq"  # "openrouter" | "groq" | "gemini"

OPENROUTER_EVALUATOR_MODEL = "poolside/laguna-s-2.1:free"
OPENROUTER_EVALUATOR_URL = OPENROUTER_API_URL
OPENROUTER_EVALUATOR_KEY = "OPENROUTER_API_KEY"
OPENROUTER_REQUEST_DELAY = 5

GROQ_EVALUATOR_MODEL = "llama-3.3-70b-versatile"
GROQ_EVALUATOR_URL = GROQ_API_URL
GROQ_EVALUATOR_KEY = "GROQ_API_KEY"
GROQ_REQUEST_DELAY = 10

GEMINI_EVALUATOR_MODEL = "gemini-2.5-flash"
GEMINI_EVALUATOR_URL = GEMINI_API_URL
GEMINI_EVALUATOR_KEY = "GEMINI_API_KEY"
GEMINI_REQUEST_DELAY = 1

# -----------------------------------------------------------------------
# Agent LLM — switch AGENT_PROVIDER to change agent model
# -----------------------------------------------------------------------
AGENT_PROVIDER = "groq"  # "openrouter" | "groq" | "gemini"

GROQ_AGENT_MODEL = "llama-3.3-70b-versatile"
OPENROUTER_AGENT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
GEMINI_AGENT_MODEL = "gemini-2.5-flash"


# -----------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------
# REWRITE_SYSTEM = """You are a search query optimizer.
# Your job is to rewrite a user's conversational question into a precise, self-contained search query.

# Rules:
# - Remove conversational filler (what did you say, can you tell me, etc.)
# - Expand abbreviations and resolve ambiguous pronouns
# - Make the query document-shaped — use the terminology that would appear in an answer
# - Return ONLY the rewritten query. No explanation. No punctuation at the end."""

REWRITE_SYSTEM = """You are a search query optimizer for a document retrieval system.
Your job is to rewrite a user's conversational question into a concise search query.

Rules:
- Keep the query SHORT — 5 to 10 words maximum
- Use only terms and concepts present in the original question
- Do NOT add information, assumptions, or domain knowledge not in the question
- Do NOT expand with related concepts or best practices
- Remove conversational filler only
- Return ONLY the rewritten query. No explanation. No punctuation at the end."""

SYSTEM_PROMPT = """You are a precise question-answering assistant.
You answer questions strictly based on the provided context.
If the context does not contain enough information to answer the question, 
say "I don't have enough information in my knowledge base to answer this."
Do not use any knowledge outside the provided context."""
