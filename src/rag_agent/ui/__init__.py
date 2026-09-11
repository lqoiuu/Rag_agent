"""Streamlit product UI: chat, knowledge base and retrieval debugging.

The pages are thin by design. Intent routing, the confirmation gate, idempotent
ingestion and citation validation all stay in ``agent/``, ``tools/``, ``memory/`` and
``ingestion/``, so the UI and the CLI exercise the same rules and the interface cannot
become a way around them.
"""

from rag_agent.ui.services import AppResources, build_resources, get_resources

__all__ = ["AppResources", "build_resources", "get_resources"]
