# This script evaluates the performance of a RAG (Retrieval-Augmented Generation) system
# using a "LLM-as-Judge" approach. It calculates several key metrics to assess the quality
# of the generated answers and the retrieved contexts.
#
# The evaluation metrics are inspired by the RAGAS framework and include:
# - Faithfulness: Measures if the answer is factually consistent with the provided context.
# - Context Precision: Measures if the retrieved context is relevant and noise-free.
# - Context Recall: Measures if the retrieved context contains all the information needed
#   to answer the question.
# - Answer Relevancy: Measures how relevant the answer is to the question.
#
# The script is designed to be modular and supports different LLM providers (e.g., OpenRouter,
# Groq, Gemini) for the evaluation. It reads the RAG system's outputs from a JSON file,
# calculates the scores for each record, and saves the results to another JSON file.
# It also includes a resume feature to continue an interrupted evaluation run.
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
# Add project root to the Python path to allow importing from config.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from config import (
    EVAL_PROVIDER,
    OPENROUTER_EVALUATOR_MODEL,
    OPENROUTER_EVALUATOR_URL,
    OPENROUTER_EVALUATOR_KEY,
    OPENROUTER_REQUEST_DELAY,
    GROQ_EVALUATOR_MODEL,
    GROQ_API_URL,
    GROQ_EVALUATOR_KEY,
    GROQ_REQUEST_DELAY,
    GEMINI_EVALUATOR_MODEL,
    GEMINI_API_URL,
    GEMINI_EVALUATOR_KEY,
    GEMINI_REQUEST_DELAY,
    EMBEDDINGS_MODEL_NAME
)

# Initialize the sentence transformer model for embedding calculation.
# This is used for the 'answer_relevancy' metric.
_embedder = SentenceTransformer(EMBEDDINGS_MODEL_NAME)
# print(f"[EMBEDDER] loaded: {EMBEDDINGS_MODEL_NAME}")    # Debug Statement

# Define paths for input results and output scores files.
RESULTS_PATH = Path(__file__).parent / "eval_results.json"
SCORES_PATH = Path(__file__).parent / "ragas_scores.json"

# --- Provider Configuration ---
# This section sets up the configuration for the LLM provider that will be used
# as the "judge" for the evaluation metrics. It allows switching between different
# providers like OpenRouter, Groq, or Gemini based on the EVAL_PROVIDER environment variable.

# A dictionary to hold the configuration details for each supported provider.
# Each tuple contains: (model_name, api_url, api_key_env_var, request_delay)
_PROVIDERS = {
    "openrouter": (
        OPENROUTER_EVALUATOR_MODEL,
        OPENROUTER_EVALUATOR_URL,
        OPENROUTER_EVALUATOR_KEY,
        OPENROUTER_REQUEST_DELAY,
    ),
    "groq": (
        GROQ_EVALUATOR_MODEL,
        GROQ_API_URL,
        GROQ_EVALUATOR_KEY,
        GROQ_REQUEST_DELAY,
    ),
    "gemini": (
        GEMINI_EVALUATOR_MODEL,
        GEMINI_API_URL,
        GEMINI_EVALUATOR_KEY,
        GEMINI_REQUEST_DELAY,
    ),
}

# Validate that the specified EVAL_PROVIDER is a valid choice.
if EVAL_PROVIDER not in _PROVIDERS:
    raise ValueError(
        f"Unknown EVAL_PROVIDER '{EVAL_PROVIDER}'. Choose: openrouter | groq | gemini"
    )

# Resolve and set the active provider's configuration at startup for use throughout the script.
ACTIVE_MODEL, ACTIVE_URL, ACTIVE_KEY_ENV, ACTIVE_DELAY = _PROVIDERS[EVAL_PROVIDER]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_evaluator(prompt: str) -> str:
    # Sends a prompt to the configured evaluation LLM and returns the response.
    #
    # This function handles the API call to the selected LLM provider (e.g., OpenRouter,
    # Groq, Gemini). It constructs the request payload, sends it, and extracts the
    # content from the response.
    #
    # Args:
    #     prompt: The prompt string to send to the LLM.
    #
    # Returns:
    #     The content of the LLM's response as a string.
    #
    # Raises:
    #     RuntimeError: If the API response is unexpected or lacks the 'choices' key.
    headers = {
        "Authorization": f"Bearer {os.getenv(ACTIVE_KEY_ENV)}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": ACTIVE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # Set to 0 for deterministic, reproducible outputs
        "max_tokens": 1024,
    }
    response = requests.post(ACTIVE_URL, headers=headers, json=payload, timeout=60)
    data = response.json()

    # Basic error handling for the API response structure.
    if "choices" not in data:
        raise RuntimeError(f"Unexpected response from {EVAL_PROVIDER}: {data}")

    return data["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    # Calculates the faithfulness score for a given answer.
    #
    # Faithfulness measures whether the claims made in the answer are supported by the
    # provided context. The LLM is prompted to break the answer into atomic claims and
    # verify each one against the context.
    #
    # Score = (Number of supported claims) / (Total number of claims)
    #
    # Args:
    #     question: The original question (for context, though not used in the prompt).
    #     answer: The generated answer to be evaluated.
    #     contexts: A list of context strings retrieved by the RAG system.
    #
    # Returns:
    #     A faithfulness score between 0.0 and 1.0. Returns 0.0 on parsing failure.
    # Combine all context chunks into a single block for the prompt.
    context_block = "\n\n".join(f"[C{i+1}] {c}" for i, c in enumerate(contexts))

    # The prompt instructs the LLM to act as an evaluator and return a structured JSON.
    prompt = f"""You are evaluating whether an answer is faithful to the given context.

Context:
{context_block}

Answer: {answer}

Instructions:
1. Break the answer into individual atomic claims.
2. For each claim, judge whether it is supported by the context.
3. Return ONLY a JSON object in this format:
{{"total": <int>, "supported": <int>}}
No explanation. No markdown."""

    raw = call_evaluator(prompt)
    # Respect the API rate limits by waiting for the configured delay.
    time.sleep(ACTIVE_DELAY)

    try:
        # Clean up the raw response to extract the JSON object.
        # This handles cases where the LLM wraps the JSON in markdown code blocks.
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(clean)
        # Calculate the score. If there are no claims, faithfulness is 1.0 (vacuously true).
        return round(data["supported"] / data["total"], 4) if data["total"] > 0 else 1.0
    except Exception:
        # If JSON parsing fails or the LLM response is malformed, return a score of 0.
        return 0.0


def score_context_precision(
    question: str, ground_truth: str, contexts: list[str]
) -> float:
    # Calculates the context precision score.
    #
    # Context precision measures how relevant the retrieved context chunks are to the
    # question and expected answer. It helps identify if the retrieval system is
    # returning noisy or irrelevant information.
    #
    # Score = (Number of relevant chunks) / (Total number of chunks)
    #
    # Args:
    #     question: The original question.
    #     ground_truth: The "gold standard" or expected answer.
    #     contexts: A list of context strings retrieved by the RAG system.
    #
    # Returns:
    #     A context precision score between 0.0 and 1.0. Returns 0.0 on failure or if no contexts.
    if not contexts:
        return 0.0

    chunks_block = "\n\n".join(f"[C{i+1}] {c}" for i, c in enumerate(contexts))

    # The prompt asks the LLM to judge the usefulness of each chunk for answering the question.
    prompt = f"""You are evaluating whether retrieved context chunks are relevant.

Question: {question}
Expected answer: {ground_truth}

Retrieved chunks:
{chunks_block}

Instructions:
For each chunk, judge whether it is useful for producing the expected answer.
Return ONLY a JSON object in this format:
{{"total": <int>, "relevant": <int>}}
No explanation. No markdown."""

    raw = call_evaluator(prompt)
    time.sleep(ACTIVE_DELAY)

    try:
        # Clean and parse the JSON response from the LLM.
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(clean)
        # Calculate the precision score.
        return round(data["relevant"] / data["total"], 4) if data["total"] > 0 else 0.0
    except Exception:
        # If parsing fails, return a score of 0.
        return 0.0


def score_context_recall(ground_truth: str, contexts: list[str]) -> float:
    # Calculates the context recall score.
    #
    # Context recall measures whether the retrieved contexts contain all the information
    # needed to formulate the expected answer (ground_truth). The LLM is asked to check
    # if each sentence of the ground truth can be inferred from the contexts.
    #
    # Score = (Number of covered sentences) / (Total number of sentences in ground truth)
    #
    # Args:
    #     ground_truth: The "gold standard" or expected answer.
    #     contexts: A list of context strings retrieved by the RAG system.
    #
    # Returns:
    #     A context recall score between 0.0 and 1.0. Returns 0.0 on failure.
    # Combine all context chunks into a single block for the prompt.
    context_block = "\n\n".join(f"[C{i+1}] {c}" for i, c in enumerate(contexts))

    # The prompt instructs the LLM to check if the ground truth is covered by the context.
    prompt = f"""You are evaluating whether retrieved contexts cover the expected answer.

Expected answer: {ground_truth}

Retrieved contexts:
{context_block}

Instructions:
1. Split the expected answer into individual sentences.
2. For each sentence, judge whether it can be inferred from the retrieved contexts.
3. Return ONLY a JSON object in this format:
{{"total": <int>, "covered": <int>}}
No explanation. No markdown."""

    raw = call_evaluator(prompt)
    time.sleep(ACTIVE_DELAY)

    try:
        # Clean and parse the JSON response.
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(clean)
        # Calculate the recall score. If no sentences, recall is 1.0.
        return round(data["covered"] / data["total"], 4) if data["total"] > 0 else 1.0
    except Exception:
        # If parsing fails, return a score of 0.
        return 0.0


def score_answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    # Calculates the answer relevancy score.
    #
    # This metric assesses how relevant the generated answer is to the original question.
    # It works by:
    # 1. Using an LLM to generate 'n' synthetic questions that the answer could be responding to.
    # 2. Embedding the original question and the synthetic questions.
    # 3. Calculating the mean cosine similarity between the original question's embedding
    #    and the synthetic questions' embeddings.
    #
    # A high score indicates that the answer is on-topic and directly addresses the question.
    #
    # Args:
    #     question: The original question.
    #     answer: The generated answer.
    #     n: The number of synthetic questions to generate.
    #
    # Returns:
    #     The mean cosine similarity score (0.0 to 1.0), or float('nan') for refusal answers.
    # If the RAG system gives a refusal answer, it's not irrelevant, but we can't
    # evaluate its relevancy in the same way. We return 'nan' so it can be excluded
    # from the final average, neither penalizing nor rewarding it.
    if "don't have enough information" in answer.lower():
        return float("nan")  # excluded from aggregate, not penalised

    # Prompt the LLM to generate synthetic questions based on the answer.
    prompt = f"""Given the following answer, generate {n} different questions that this answer could be responding to.
Return ONLY a JSON array of {n} question strings. No explanation. No markdown.

Answer: {answer}

Example output: ["What color is the sky?", "Why is the sky blue?", "What does the sky look like?"]"""

    raw = call_evaluator(prompt)
    time.sleep(ACTIVE_DELAY)

    # DEBUG — remove after fixing
    # print(f"    [AR DEBUG] raw response: {repr(raw[:200])}")

    try:
        # Clean and parse the JSON array of questions.
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        synthetic_question = json.loads(clean)
        if not isinstance(synthetic_question, list) or len(synthetic_question) == 0:
            # print(
            #     f"    [AR DEBUG] parsed but empty or wrong type: {type(synthetic_question)}"
            # )
            return 0.0

    except Exception as e:
        # If parsing fails or the response is malformed, return 0.
        # print(f"    [AR DEBUG] JSON parse failed: {e} | clean: {repr(clean[:200])}")
        return 0.0

    # Embed original question and all synthetic questions using the sentence transformer.
    # Normalizing embeddings allows using the dot product for cosine similarity.
    all_texts = [question] + synthetic_question
    embeddings = _embedder.encode(all_texts, normalize_embeddings=True)

    original_emb = embeddings[0]
    synthetic_emb = embeddings[1:]

    # Cosine similarity is the dot product when embeddings are normalised.
    similarities = [float(np.dot(original_emb, syn_emb)) for syn_emb in synthetic_emb]

    # print(f"    [AR DEBUG] synthetic_questions: {synthetic_question}")
    # print(f"    [AR DEBUG] embedding norms: {[float(np.linalg.norm(e)) for e in synthetic_emb]}") 
    # print(f"    [AR DEBUG] similarities: {similarities}")

    # The final score is the average of these similarities.
    return round(float(np.mean(similarities)), 4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # The main function to run the RAG evaluation process.
    #
    # It performs the following steps:
    # 1. Prints the active evaluation provider and model.
    # 2. Loads the RAG system's output records from `eval_results.json`.
    # 3. Filters out any records that resulted in an error during generation.
    # 4. Implements a resume feature by checking for already scored records in `ragas_scores.json`.
    # 5. Iterates through each record, calculates the four evaluation metrics
    #    (faithfulness, context precision, context recall, answer relevancy).
    # 6. Appends the new scores to the list of existing scores.
    # 7. Saves all scored records to `ragas_scores.json`.
    # 8. Calculates and prints the average scores for all metrics.
    # 9. Prints a per-question summary of the scores.
    print("=" * 60)
    print("Phase 3 — RAG Evaluation (LLM-as-Judge)")
    print(f"Provider: {EVAL_PROVIDER} — model: {ACTIVE_MODEL}")
    print("=" * 60)

    # Ensure the input file from the previous evaluation step exists.
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run run_eval.py first.")
        sys.exit(1)

    # Load the results of the RAG pipeline execution.
    with open(RESULTS_PATH) as f:
        records = json.load(f)

    # Filter out records where the RAG system failed to produce an answer.
    valid = [r for r in records if not r["answer"].startswith("ERROR")]
    # valid = [r for r in records if r["id"] == "Q01"]    # Debug statement

    print(f"\nEvaluating {len(valid)} records...\n")

    # --- Resume Support ---
    # This allows the script to be stopped and restarted without re-calculating
    # scores that have already been computed and saved.
    if SCORES_PATH.exists():
        with open(SCORES_PATH) as f:
            existing = json.load(f)
        # Create a set of IDs for quick lookup.
        already_scored = {r["id"] for r in existing}
        if already_scored:
            print(f"Resuming — {len(already_scored)} already scored: {already_scored}\n")
    else:
        existing = []
        already_scored = set()

    scored = []
    # Iterate over each valid record to calculate its scores.
    for i, r in enumerate(valid):
        qid = r["id"]

        # Skip records that have already been scored in a previous run.
        if qid in already_scored:
            print(f"[{i+1}/{len(valid)}] {qid} — skipping (already scored)")
            continue

        print(f"[{i+1}/{len(valid)}] {qid} — {r['question'][:55]}...")

        # Calculate each of the four metrics for the current record.
        faith = score_faithfulness(r["question"], r["answer"], r["contexts"])
        print(f"  faithfulness:      {faith}")

        prec = score_context_precision(r["question"], r["ground_truth"], r["contexts"])
        print(f"  context_precision: {prec}")

        rec = score_context_recall(r["ground_truth"], r["contexts"])
        print(f"  context_recall:    {rec}")

        ar = score_answer_relevancy(r["question"], r["answer"])
        print(f"  answer_relevancy:  {ar}")

        # Combine the original record with its new scores.
        scored.append(
            {
                **r,
                "faithfulness": faith,
                "context_precision": prec,
                "context_recall": rec,
                "answer_relevancy" : ar
            }
        )

    # Combine scores from this run with scores from previous runs.
    all_scored = existing + scored

    # Save the updated list of all scored records to the output file.
    with open(SCORES_PATH, "w") as f:
        json.dump(all_scored, f, indent=2)

    if not all_scored:
        print("No scores yet.")
        return

    # --- Calculate and Display Aggregate Scores ---

    # Filter out 'nan' values for answer relevancy before calculating the average.
    # This handles the refusal answers correctly.
    ar_scores = [
        s["answer_relevancy"]
        for s in all_scored
        if not (
            isinstance(s["answer_relevancy"], float) and np.isnan(s["answer_relevancy"])
        )
    ]

    # Calculate the average for each metric across all scored questions.
    avg_faith = round(sum(s["faithfulness"] for s in all_scored) / len(all_scored), 4)
    avg_prec = round(
        sum(s["context_precision"] for s in all_scored) / len(all_scored), 4
    )
    avg_rec = round(sum(s["context_recall"] for s in all_scored) / len(all_scored), 4)
    avg_ar = round(sum(ar_scores) / len(ar_scores), 4) if ar_scores else float("nan")

    # Print a summary of the aggregate scores.
    print("\n" + "=" * 60)
    print(f"SCORES SO FAR ({len(all_scored)}/{len(valid)} questions)")
    print("=" * 60)
    print(f"  Faithfulness:      {avg_faith}")
    print(f"  Context Precision: {avg_prec}")
    print(f"  Context Recall:    {avg_rec}")
    print(f"  Answer Relevancy:  {avg_ar} (excludes refusal answers)")

    # Print a detailed, per-question breakdown of the scores.
    print("\nPer-question:")
    print("-" * 60)
    for s in all_scored:
        print(
            f"  {s['id']} ({s['difficulty']:<6})  "
            f"F={s['faithfulness']:.2f}  "
            f"P={s['context_precision']:.2f}  "
            f"R={s['context_recall']:.2f}  "
            f"A={s['answer_relevancy']:.2f}  "
            f"— {s['question'][:45]}"
        )

    print(f"\nFull results saved to: {SCORES_PATH}")


# Standard Python entry point.
if __name__ == "__main__":
    main()
