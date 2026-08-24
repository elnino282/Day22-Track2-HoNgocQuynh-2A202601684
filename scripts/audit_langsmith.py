"""Read-only audit of rubric-critical LangSmith evidence."""

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config
from langsmith import Client
from prompt_versions import PROMPT_V1, PROMPT_V2  # noqa: F401 - verifies imports


PROMPT_NAMES = (
    "ho-ngoc-quynh-rag-prompt-v1",
    "ho-ngoc-quynh-rag-prompt-v2",
)


def main() -> None:
    client = Client(api_key=config.LANGSMITH_API_KEY)
    runs = list(
        client.list_runs(
            project_name=config.LANGSMITH_PROJECT,
            is_root=True,
            select=["id", "name", "error", "inputs", "outputs"],
            # LangSmith currently caps a single query at 100; this is enough
            # for the rubric's expected 50 + 50 root traces.
            limit=100,
        )
    )
    counts = Counter(run.name for run in runs)
    errors = sum(run.error is not None for run in runs)
    print(f"Project: {config.LANGSMITH_PROJECT}")
    print(f"Root runs: {len(runs)}; errors: {errors}; by name: {dict(counts)}")

    rag_runs = [run for run in runs if run.name == "rag-query"]
    complete = sum(bool(run.inputs) and bool(run.outputs) for run in rag_runs)
    print(f"rag-query: {len(rag_runs)}; có input + output: {complete}")

    for name in PROMPT_NAMES:
        prompt = client.get_prompt(name)
        print(f"Prompt {name}: {'FOUND' if prompt else 'MISSING'}")


if __name__ == "__main__":
    main()
