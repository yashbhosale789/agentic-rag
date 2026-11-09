import logging
import os
from typing import Any

# from langchain.agents import AgentExecutor
# from langchain import hub
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool as lc_tool

# from langchain.tools import Tool

from src.agent.tools import get_all_tools
from config import (
    AGENT_PROVIDER,
    GROQ_AGENT_MODEL,
    GROQ_API_KEY,
    OPENROUTER_AGENT_MODEL,
    OPENROUTER_API_URL,
    OPENROUTER_API_KEY,
    GEMINI_AGENT_MODEL,
    GEMINI_API_KEY,
    SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def _build_llm() -> Any:
    if AGENT_PROVIDER == "groq":
        logger.info(f"Agent LLM: Groq - {GROQ_AGENT_MODEL}")
        return ChatGroq(
            model=GROQ_AGENT_MODEL, api_key=os.getenv(GROQ_API_KEY), temperature=0.2
        )

    if AGENT_PROVIDER == "openrouter":
        logger.info(f"Agent LLM: Openrouter - {OPENROUTER_AGENT_MODEL}")
        return ChatOpenAI(
            model=OPENROUTER_AGENT_MODEL,
            api_key=os.getenv(OPENROUTER_API_KEY),
            base_url=OPENROUTER_API_URL.replace("/chat/completions", ""),
            temperature=0.2,
        )

    if AGENT_PROVIDER == "gemini":
        logger.info(f"Agent LLM: Gemini - {GEMINI_AGENT_MODEL}")
        return ChatGoogleGenerativeAI(
            model=GEMINI_AGENT_MODEL, api_key=os.getenv(GEMINI_API_KEY), temperature=0.2
        )

    raise ValueError(
        f"Unknown AGENT_PROVIDER: '{AGENT_PROVIDER}'. "
        "Choose 'groq', 'openrouter', or 'gemini'."
    )


def _build_langchain_tools() -> list[Any]:
    raw_tools = get_all_tools()
    langchain_tools = []

    for raw in raw_tools:
        name = raw["name"]
        description = raw["description"]
        func = raw["function"]

        def make_tool(f, n, d):
            @lc_tool(n, description=d)
            def _tool(query: str) -> str:
                return f(query)
            return _tool

        langchain_tools.append(make_tool(func, name, description))

    return langchain_tools


def run_langchain_agent(query: str) -> dict[str, Any]:
    llm = _build_llm()
    tools = _build_langchain_tools()

    # prompt = hub.pull("hwchase17/react")

    AGENT_SYSTEM_PROMPT = """You are a precise question-answering agent with access to tools.

Rules:
1. Always use the retrieve tool to find information before answering factual questions.
2. Use the get_document_info tool when the question is about what document or knowledge base exists, not about specific content within it.
3. Answer only from what the tools return. Do not use your own training knowledge.
4. If the tools do not return relevant information, say so explicitly.
5. Never make up information."""

    agent = create_agent(model=llm, tools=tools, system_prompt=AGENT_SYSTEM_PROMPT, debug=True)

    # executor = AgentExecutor(
    #     agent=agent,
    #     tools=tools,
    #     max_iterations=6,
    #     handle_parsing_errors=True,
    #     verbose=True,
    # )

    logger.info(f"Langchain agent started for: {query[:60]}")

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    final_message = result["messages"][-1]
    answer = (
        final_message.content
        if hasattr(final_message, "content")
        else str(final_message)
    )

    return {"query": query, "answer": answer}
