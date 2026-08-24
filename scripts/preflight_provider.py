"""Minimal provider connectivity check without creating LangSmith traces."""

import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import config
from utils.llm_factory import get_embeddings, get_llm


def main() -> None:
    print(f"Provider: {config.PROVIDER}; model: {config.OPENAI_MODEL}", flush=True)

    started = time.perf_counter()
    vector = get_embeddings().embed_query("Retrieval-Augmented Generation")
    print(
        f"Embeddings OK: dimension={len(vector)}, elapsed={time.perf_counter() - started:.2f}s",
        flush=True,
    )

    started = time.perf_counter()
    reply = get_llm().invoke("Reply with exactly: PROVIDER_OK")
    content = getattr(reply, "content", str(reply))
    print(
        f"LLM OK: {content!r}, elapsed={time.perf_counter() - started:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
