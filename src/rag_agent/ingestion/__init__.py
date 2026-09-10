"""Document ingestion: decoding, normalization, loading and chunking."""

from rag_agent.ingestion.loaders import (
    LoadBatch,
    LoadFailure,
    iter_supported_files,
    load_document,
    load_documents,
    source_name,
)
from rag_agent.ingestion.normalize import decode_bytes, is_blank, normalize_text
from rag_agent.ingestion.splitters import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ChunkingConfig,
    TextSection,
    TextSplitter,
    build_langchain_splitter,
    split_document,
    split_into_sections,
)
from rag_agent.ingestion.stats import ChunkStats, summarize_chunks

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "ChunkStats",
    "ChunkingConfig",
    "LoadBatch",
    "LoadFailure",
    "TextSection",
    "TextSplitter",
    "build_langchain_splitter",
    "decode_bytes",
    "is_blank",
    "iter_supported_files",
    "load_document",
    "load_documents",
    "normalize_text",
    "source_name",
    "split_document",
    "split_into_sections",
    "summarize_chunks",
]
