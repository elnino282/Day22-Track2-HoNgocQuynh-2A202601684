"""
Tải cấu hình từ file .env và thiết lập biến môi trường LangSmith.

⚠️  Import module này TRƯỚC KHI import bất kỳ thư viện LangChain nào.
    config.py tự động set LANGCHAIN_* vào os.environ khi được import.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from utils.console import configure_utf8_console

configure_utf8_console()

# Tải .env từ thư mục gốc của project (Lab/)
_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

# ── LangSmith — PHẢI set trước khi import LangChain ──────────────────────
_langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
_langsmith_project = os.getenv("LANGSMITH_PROJECT") or os.getenv(
    "LANGCHAIN_PROJECT", "day22-lab"
)
_langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)
_tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2") or os.getenv(
    "LANGSMITH_TRACING", "true"
)

# Export both current LANGSMITH_* and legacy LANGCHAIN_* names. This keeps the
# lab compatible with the Guide and multiple LangSmith SDK generations.
os.environ["LANGCHAIN_TRACING_V2"] = _tracing_enabled
os.environ["LANGSMITH_TRACING"] = _tracing_enabled
os.environ["LANGCHAIN_API_KEY"] = _langsmith_key
os.environ["LANGSMITH_API_KEY"] = _langsmith_key
os.environ["LANGCHAIN_PROJECT"] = _langsmith_project
os.environ["LANGSMITH_PROJECT"] = _langsmith_project
os.environ["LANGCHAIN_ENDPOINT"] = _langsmith_endpoint
os.environ["LANGSMITH_ENDPOINT"] = _langsmith_endpoint

# ── Provider mặc định ─────────────────────────────────────────────────────
# Đổi giá trị PROVIDER trong .env: openai | gemini | anthropic | ollama | openrouter
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# ── OpenAI ────────────────────────────────────────────────────────────────
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL        = os.getenv("OPENAI_BASE_URL", "")   # để trống nếu dùng OpenAI chính thức
OPENAI_MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
REQUEST_TIMEOUT        = float(os.getenv("REQUEST_TIMEOUT", "60"))
MAX_RETRIES            = int(os.getenv("MAX_RETRIES", "2"))
MAX_WORKERS            = max(1, int(os.getenv("MAX_WORKERS", "4")))

# RAGAS evaluator calls are longer and heavier than normal RAG generation.
# Keep their limits separate so the proxy is not overloaded by four concurrent
# judge calls or cut off at the normal 60-second request timeout.
RAGAS_TIMEOUT          = float(os.getenv("RAGAS_TIMEOUT", "240"))
RAGAS_MAX_RETRIES      = int(os.getenv("RAGAS_MAX_RETRIES", "4"))
RAGAS_MAX_WORKERS      = max(1, int(os.getenv("RAGAS_MAX_WORKERS", "2")))

# ── Google Gemini ─────────────────────────────────────────────────────────
GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL            = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL  = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

# ── Anthropic ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# ── Ollama (local, không cần API key) ────────────────────────────────────
OLLAMA_BASE_URL         = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL            = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBEDDING_MODEL  = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ── OpenRouter ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── LangSmith ─────────────────────────────────────────────────────────────
LANGSMITH_API_KEY = _langsmith_key
LANGSMITH_PROJECT = _langsmith_project


def _is_configured(value: str) -> bool:
    """Reject empty values and the obvious placeholders from .env.example."""
    normalized = value.strip().lower()
    return bool(normalized) and not any(
        marker in normalized for marker in ("your_", "...", "changeme")
    )


def validate() -> bool:
    """
    Kiểm tra các biến môi trường bắt buộc đã được cấu hình.
    Trả về True nếu hợp lệ, False nếu thiếu.
    """
    missing = []

    supported = {"openai", "gemini", "anthropic", "ollama", "openrouter"}
    if PROVIDER not in supported:
        print(f"❌ PROVIDER không hợp lệ: '{PROVIDER}'. Chọn: {', '.join(sorted(supported))}")
        return False

    if _tracing_enabled.lower() != "true":
        missing.append("LANGCHAIN_TRACING_V2=true")

    if not _is_configured(LANGSMITH_API_KEY):
        missing.append("LANGSMITH_API_KEY (hoặc LANGCHAIN_API_KEY)")

    if PROVIDER == "openai" and not _is_configured(OPENAI_API_KEY):
        missing.append("OPENAI_API_KEY")
    elif PROVIDER == "gemini" and not _is_configured(GOOGLE_API_KEY):
        missing.append("GOOGLE_API_KEY")
    elif PROVIDER == "anthropic" and not _is_configured(ANTHROPIC_API_KEY):
        missing.append("ANTHROPIC_API_KEY")
        if not _is_configured(OPENAI_API_KEY):
            missing.append("OPENAI_API_KEY (embeddings cho Anthropic)")
    elif PROVIDER == "openrouter" and not _is_configured(OPENROUTER_API_KEY):
        missing.append("OPENROUTER_API_KEY")
    elif PROVIDER == "openrouter" and not _is_configured(OPENAI_API_KEY):
        missing.append("OPENAI_API_KEY (embeddings cho OpenRouter)")
    # Ollama: không cần API key

    if missing:
        print("⚠️  Thiếu biến môi trường:")
        for m in missing:
            print(f"   - {m}")
        print("   Hãy kiểm tra file .env của bạn (xem .env.example để biết thêm).")
        return False

    print(f"✅ Config OK  |  Provider: {PROVIDER.upper()}  |  Project: {LANGSMITH_PROJECT}")
    return True


if __name__ == "__main__":
    validate()
