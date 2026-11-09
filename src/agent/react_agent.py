"""ReAct agent as a LangGraph StateGraph.

    START -> agent --(Action)--> tools --(retrieve returned chunks)--> grade -> agent
               |                   '--(other tool / nothing found)-----------> agent
               '--(Answer, unparseable output, or MAX_ITERATIONS)--> END

agent: one LLM step of the Thought / Action / Answer text protocol.
tools: runs the chosen tool and appends the Observation.
grade: one LLM call judges whether the retrieved chunks are relevant; if not, a note tells the
       agent to search again with another query (corrective RAG) or say the knowledge base lacks it.
"""

import logging
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.prompts import GRADE_SYSTEM, build_system_prompt, format_observation
from src.agent.tools import get_all_tools
from src.llm.client import call_llm as get_llm_response

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6

NOT_RELEVANT_NOTE = (
    "Grader: these chunks do not look relevant to the question. Search again with a different "
    "query, or answer that the knowledge base does not contain the information."
)


class AgentState(TypedDict):
    query: str
    transcript: str  # the ReAct text the agent LLM reads at every step
    trajectory: list[dict[str, Any]]  # one entry per agent step, plus "grade" after a retrieve
    iterations: int
    observation: str  # the last tool output


def _parse_llm_response(response: str) -> dict[str, Any]:
    response = response.strip()

    if "Answer:" in response:
        answer_match = re.search(r"Answer:\s*(.+)", response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else response
        return {"type": "answer", "content": answer}

    if "Action:" in response and "Action Input:" in response:
        action_match = re.search(r"Action:\s*(.+)", response)
        input_match = re.search(r"Action Input:\s*(.+)", response)
        if action_match and input_match:
            return {
                "type": "action",
                "action": action_match.group(1).strip(),
                "action_input": input_match.group(1).strip(),
            }

    logger.warning(f"Could not parse LLM response:\n{response}")
    return {"type": "unknown", "content": response}


def _find_tool(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool in tools:
        if tool["name"].lower() == name.lower():
            return tool
    return None


def build_graph(tools: list[dict[str, Any]]) -> Any:
    system_prompt = build_system_prompt(tools)

    def agent(state: AgentState) -> dict[str, Any]:
        n = state["iterations"] + 1
        logger.info(f"--- Iteration {n} ---")
        response = get_llm_response(prompt=state["transcript"], system=system_prompt)
        logger.info(f"LLM response:\n{response}")
        step = {"iteration": n, "raw": response, "parsed": _parse_llm_response(response)}
        return {
            "iterations": n,
            "trajectory": state["trajectory"] + [step],
            "transcript": state["transcript"] + response + "\n\n",
        }

    def after_agent(state: AgentState) -> str:
        if state["trajectory"][-1]["parsed"]["type"] != "action":
            return END
        if state["iterations"] >= MAX_ITERATIONS:
            return END  # the last action's observation would never be read
        return "tools"

    def run_tool(state: AgentState) -> dict[str, Any]:
        parsed = state["trajectory"][-1]["parsed"]
        tool = _find_tool(tools, parsed["action"])
        if tool is None:
            observation = f"Error: tool '{parsed['action']}' does not exist."
            logger.warning(f"Agent called unknown tool: '{parsed['action']}'")
        else:
            observation = tool["function"](parsed["action_input"])
        return {
            "observation": observation,
            "transcript": state["transcript"] + format_observation(observation) + "\n\n",
        }

    def after_tool(state: AgentState) -> str:
        # "[CHUNK" marks real chunks (src/agent/tools.py); no-result and error texts skip the grader
        is_retrieve = state["trajectory"][-1]["parsed"]["action"].lower() == "retrieve"
        return "grade" if is_retrieve and state["observation"].startswith("[CHUNK") else "agent"

    def grade(state: AgentState) -> dict[str, Any]:
        verdict = get_llm_response(
            prompt=f"Question: {state['query']}\n\n{state['observation']}", system=GRADE_SYSTEM
        )
        relevant = verdict.strip().lower().startswith("yes")
        logger.info(f"Grader: {'relevant' if relevant else 'not relevant'} ({verdict!r})")
        trajectory = state["trajectory"]
        update: dict[str, Any] = {
            "trajectory": trajectory[:-1] + [{**trajectory[-1], "grade": "yes" if relevant else "no"}]
        }
        if not relevant:
            update["transcript"] = state["transcript"] + NOT_RELEVANT_NOTE + "\n\n"
        return update

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tool)
    graph.add_node("grade", grade)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", END])
    graph.add_conditional_edges("tools", after_tool, ["grade", "agent"])
    graph.add_edge("grade", "agent")
    return graph.compile()


def run_agent(query: str) -> dict[str, Any]:
    final = build_graph(get_all_tools()).invoke(
        {
            "query": query,
            "transcript": f"User Question: {query}\n\n",
            "trajectory": [],
            "iterations": 0,
            "observation": "",
        },
        # up to 3 nodes per iteration; LangGraph's default limit of 25 would cut off MAX_ITERATIONS >= 9
        {"recursion_limit": 3 * MAX_ITERATIONS + 1},
    )

    last = final["trajectory"][-1]["parsed"]
    if last["type"] == "answer":
        answer = last["content"]
    elif last["type"] == "unknown":
        logger.error("Agent produced unparseable output. Stopping.")
        answer = "I was unable to produce a properly formatted answer."
    else:
        logger.warning("Agent reached max iterations without answering.")
        answer = "I reached the maximum number of retrieval steps without finding a complete answer."

    return {
        "query": query,
        "answer": answer,
        "trajectory": final["trajectory"],
        "iterations": final["iterations"],
    }
