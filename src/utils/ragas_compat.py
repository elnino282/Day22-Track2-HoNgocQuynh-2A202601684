"""Compatibility helpers for the RAGAS 0.4 / LangChain 1.x combination."""

import importlib
import sys
import types


def install_vertexai_import_shim() -> None:
    """Provide a removed optional VertexAI symbol expected by RAGAS 0.4.3.

    RAGAS imports ``langchain_community.chat_models.vertexai.ChatVertexAI``
    unconditionally, although LangChain Community 0.4 removed that optional
    module. The lab does not use VertexAI. A marker class is enough for RAGAS'
    later ``isinstance`` capability check and leaves all supported providers
    untouched.
    """
    module_name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    compatibility_module = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - marker for an unused integration
        """Compatibility marker for RAGAS' multiple-completion type check."""

    ChatVertexAI.__module__ = module_name
    compatibility_module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = compatibility_module
