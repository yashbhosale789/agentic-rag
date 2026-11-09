import requests
import os
import logging
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

from config import (
    PIPELINE_PROVIDER,
    OPENROUTER_API_URL,
    OPENROUTER_MODEL,
    OPENROUTER_API_KEY,
    GROQ_API_URL,
    GROQ_PIPELINE_MODEL,
    GROQ_API_KEY,
    GEMINI_API_URL,
    GEMINI_PIPELINE_MODEL,
    GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)

_PROVIDERS = {
    "openrouter": (OPENROUTER_MODEL, OPENROUTER_API_URL, OPENROUTER_API_KEY),
    "groq": (GROQ_PIPELINE_MODEL, GROQ_API_URL, GROQ_API_KEY),
    "gemini": (GEMINI_PIPELINE_MODEL, GEMINI_API_URL, GEMINI_API_KEY),
}

if PIPELINE_PROVIDER not in _PROVIDERS:
    raise ValueError(
        f"Unknown PIPELINE_PROVIDER '{PIPELINE_PROVIDER}'. "
        "Choose: openrouter | groq | gemini"
    )

ACTIVE_MODEL, ACTIVE_URL, ACTIVE_KEY_ENV = _PROVIDERS[PIPELINE_PROVIDER]
logger.info(f"Pipeline LLM: {PIPELINE_PROVIDER} — {ACTIVE_MODEL}")


def call_llm(prompt: str, system: str = "") -> str:
    """
    Send a prompt to the active pipeline LLM and return the response text.
    Provider is controlled by PIPELINE_PROVIDER in config.py.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": ACTIVE_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {os.getenv(ACTIVE_KEY_ENV)}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/personal-kb-rag",
    }

    try:
        response = requests.post(ACTIVE_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        logger.error(f"{PIPELINE_PROVIDER} request timeout")
        raise

    except requests.exceptions.HTTPError as e:
        logger.error(
            f"{PIPELINE_PROVIDER} HTTP error: {e.response.status_code} — {e.response.text}"
        )
        raise

    except KeyError:
        logger.error(f"Unexpected response format: {data}")
        raise
