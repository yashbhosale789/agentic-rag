# Agentic RAG

An LLM agent decides when and what to search: it writes its own search queries, can search more than once, and answers from what the searches return. One of six RAG variants in a portfolio (Naive, Advanced, Production, Agentic, Graph, Multimodal) measured on the same corpus and dataset; the production system is [Personal-Knowledge-RAG](https://github.com/yashbhosale789/Personal-Knowledge-RAG).

## Design

Two agents over the same tools; the measured one is the hand-written ReAct agent.

| Part | Choice |
|---|---|
| ReAct agent (`src/agent/react_agent.py`) | Text protocol `Thought / Action / Action Input / Observation / Answer`, parsed with regexes; one tool call per step |
| Stop condition | an `Answer:` line, unparseable output, or 6 iterations (`MAX_ITERATIONS`) |
| LangChain agent (`src/agent/langchain_agent.py`) | `langchain.agents.create_agent` (LangGraph tool calling), same tools; not measured |
| Tool `retrieve` | LLM query rewrite → dense + BM25 → reciprocal rank fusion → cross-encoder rerank → top 5 chunk texts |
| Tool `get_document_info` | document and chunk counts and the corpus domain, from `storage/chunks.json` (no LLM) |
| LLM | Groq `llama-3.3-70b-versatile`, temperature 0.2, max 512 tokens (`PIPELINE_PROVIDER` / `AGENT_PROVIDER` in `config.py`); OpenRouter and Gemini are alternatives |

LLM calls per question: one per agent step (at least 2: search, then answer) plus one rewrite per `retrieve` call, so at least 3 and at most 12 (6 steps that all search, then stop without an answer).

Retrieval stack (identical to the Advanced RAG variant):

| Stage | Choice |
|---|---|
| Loading | LangChain `PyPDFLoader` (one document per page); `.txt`, `.md`, `.docx` also read |
| Chunking | `RecursiveCharacterTextSplitter`, 512 characters, 64 overlap; chunks also saved to `storage/chunks.json` for BM25 |
| Embedding | `BAAI/bge-small-en-v1.5` (384 dims, local), query prefix `Represent this sentence: ` |
| Retrieval | dense (Chroma) top 20 + BM25 (`rank-bm25`, whitespace tokens) top 20 |
| Fusion | reciprocal rank fusion, k = 60, top 10 |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 5 |

### Compared with Production RAG

Production RAG runs a fixed pipeline: one rewrite, hybrid retrieval with 1024-character chunks and PDF text clean-up, rerank to top 3, a sufficiency check that can abstain, generation with citations and a citation check. This variant lets the model choose the query and the number of searches, has no abstention gate or citation check, and uses 512-character chunks without text clean-up. Production RAG keeps the same two agents as unused foundations (`src/agent/`); they are not wired into its API.

## Run

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
echo GROQ_API_KEY=your_key > .env                # OPENROUTER_API_KEY / GEMINI_API_KEY for other providers

# Index a corpus folder in place (default: data/documents). Delete storage/ before re-indexing:
# chunks get random ids, so a second ingest duplicates them in Chroma.
RAG_DATA_DIR=/path/to/corpus RAG_DATA_GLOB="*.pdf" python scripts/ingest.py
python scripts/agentic_query.py      # asks for the agent (1 ReAct, 2 LangChain) and a question
```

Evaluation (dataset: `eval/data/golden_dataset.json` of Personal-Knowledge-RAG, read by path):

```bash
python scripts/retrieval_eval.py <golden_dataset.json>                    # $0: the retrieve tool on the raw question
python scripts/retrieval_eval.py <golden_dataset.json> --agent --limit 5  # LLM calls: the agent, 5 questions
python scripts/retrieval_eval.py <golden_dataset.json> --agent            # the rest; re-run to resume after a 429
python scripts/compare.py eval/agent_results.json:agent_first <prod_results.json>:reranked recall@3 --family 6
```

`eval/retrieval_results.json` holds the $0 run (stages dense, bm25, fused, reranked), `eval/agent_results.json` the agent run: per question the first search's top 5 (`agent_first`), everything the agent retrieved (`agent_all`, recall@all), the tool calls and queries, LLM calls and tokens, latency and the answer.

## Results (provisional)

Not measured yet. Metrics: recall@1/3/5, success@k and MRR on the 71 answerable questions of the Personal-Knowledge-RAG golden dataset (id `8f89d42be9`), with the metric definitions of that project's eval harness: a hit means the labelled evidence span (source file + text, chunk-independent) is inside a retrieved chunk. **Provisional:** every variant is re-measured through one shared eval kit before the final cross-variant table.

## Corpus and licence

The measurement corpus is [RAG-Multi-Corpus](https://github.com/udayallu/RAG-Multi-Corpus) (commit `39071af3f4dd25e59f5c59a6f9b6e8e99cd643b3`): every PDF under `datasets/<Company>/pdf/`, 236 files, synthetic documents of five fictional companies. It is MIT-licensed, Copyright (c) 2025 UDAY ALLU. This repository ships one of its files as a sample (`data/documents/`, with the licence in `data/documents/LICENSE`); download the rest from the upstream repository.
