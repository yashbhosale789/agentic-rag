import logging
import re
from typing import Any

from src.agent.prompts import build_system_prompt, format_observation
from src.agent.tools import get_all_tools
from src.llm.client import call_llm as get_llm_response

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6


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


def run_agent(query: str) -> dict[str, Any]:
    tools = get_all_tools()
    system_prompt = build_system_prompt(tools)

    transcript = f"User Question: {query}\n\n"

    iterations = 0
    trajectory = []

    while iterations < MAX_ITERATIONS:
        iterations += 1
        logger.info(f"--- Iteration {iterations} ---")

        response = get_llm_response(prompt=transcript, system=system_prompt)

        logger.info(f"LLM response:\n{response}")
        parsed = _parse_llm_response(response)
        trajectory.append({"iteration": iterations, "raw": response, "parsed": parsed})

        if parsed["type"] == "answer":
            return {
                "query": query,
                "answer": parsed["content"],
                "trajectory": trajectory,
                "iterations": iterations,
            }

        if parsed["type"] == "action":
            action_name = parsed["action"]
            action_input = parsed["action_input"]

            tool = _find_tool(tools, action_name)
            if tool is None:
                observation = f"Error: tool '{action_name}' does not exist."
                logger.warning(f"Agent called unknown tool: '{action_name}'")
            else:
                observation = tool["function"](action_input)

            transcript += response + "\n\n"
            transcript += format_observation(observation) + "\n\n"
            continue

        logger.error("Agent produced unparseable output. Stopping.")
        return {
            "query": query,
            "answer": "I was unable to produce a properly formatted answer.",
            "trajectory": trajectory,
            "iterations": iterations,
        }

    logger.warning("Agent reached max iterations without answering.")
    return {
        "query": query,
        "answer": "I reached the maximum number of retrieval steps without finding a complete answer.",
        "trajectory": trajectory,
        "iterations": iterations,
    }