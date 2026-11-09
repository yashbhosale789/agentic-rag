"""LangGraph agent routing with the LLM and tools stubbed: no API calls, no index needed."""

import sys
import types

# the real tools module builds BM25 from storage/chunks.json and loads the cross-encoder on import
sys.modules.setdefault("src.agent.tools", types.SimpleNamespace(get_all_tools=list))

import src.agent.react_agent as agent  # noqa: E402
from src.agent.prompts import GRADE_SYSTEM  # noqa: E402

SEARCH = "Thought: search\nAction: retrieve\nAction Input: tyre change steps"


def _run(monkeypatch, replies: list[str], chunks: str = "[CHUNK 1]\nLoosen the wheel nuts."):
    systems, queries = [], []

    def llm(prompt: str, system: str = "") -> str:
        systems.append(system)
        return replies.pop(0)

    def retrieve(query: str) -> str:
        queries.append(query)
        return chunks

    monkeypatch.setattr(agent, "get_llm_response", llm)
    monkeypatch.setattr(agent, "get_all_tools", lambda: [{"name": "retrieve", "description": "d", "function": retrieve}])
    return agent.run_agent("How do I change a tyre?"), systems, queries


def test_grader_no_sends_agent_back_to_search(monkeypatch):
    result, systems, queries = _run(monkeypatch, [SEARCH, "no", SEARCH, "yes", "Thought: done\nAnswer: Loosen the nuts."])
    assert result["answer"] == "Loosen the nuts."
    assert result["iterations"] == 3
    assert [s.get("grade") for s in result["trajectory"]] == ["no", "yes", None]
    assert systems.count(GRADE_SYSTEM) == 2
    assert len(queries) == 2


def test_empty_retrieval_skips_grader(monkeypatch):
    result, systems, _ = _run(monkeypatch, [SEARCH, "Thought: nothing\nAnswer: Not in the knowledge base."],
                              chunks="No relevant information found in the document.")
    assert result["answer"] == "Not in the knowledge base."
    assert GRADE_SYSTEM not in systems


def test_stops_at_max_iterations(monkeypatch):
    result, _, queries = _run(monkeypatch, [SEARCH, "yes"] * agent.MAX_ITERATIONS)
    assert result["iterations"] == agent.MAX_ITERATIONS
    assert result["trajectory"][-1]["parsed"]["type"] == "action"
    assert "maximum number" in result["answer"]
    assert len(queries) == agent.MAX_ITERATIONS - 1  # the last action is not run


def test_unparseable_output_stops(monkeypatch):
    result, _, _ = _run(monkeypatch, ["I am not following the format"])
    assert result["iterations"] == 1
    assert "unable" in result["answer"]
