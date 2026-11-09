import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.react_agent import run_agent
from src.agent.langchain_agent import run_langchain_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)

logger = logging.getLogger(__name__)


def print_trajectory(trajectory: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("AGENT TRAJECTORY")
    print("=" * 60)
    for step in trajectory:
        print(f"\n--- Iteration {step['iteration']} ---")
        print(step["raw"])
    print("\n" + "=" * 60)


def main() -> None:
    print("Select agent:")
    print("  1 — Manual ReAct agent")
    print("  2 — LangChain AgentExecutor")
    choice = input("Enter 1 or 2: ").strip()

    if choice not in ("1", "2"):
        print("Invalid choice. Exiting.")
        sys.exit(1)

    query = input("Enter your query: ").strip()
    if not query:
        print("No query entered. Exiting.")
        sys.exit(1)

    print(f"\nQuery: {query}")
    print("-" * 60)

    if choice == "1":
        result = run_agent(query)
        print_trajectory(result["trajectory"])
        print("\nFINAL ANSWER")
        print("=" * 60)
        print(result["answer"])
        print(f"\nCompleted in {result['iterations']} iteration(s).")

    if choice == "2":
        result = run_langchain_agent(query)
        print("\nFINAL ANSWER")
        print("=" * 60)
        print(result["answer"])

if __name__ == "__main__":
    main()
