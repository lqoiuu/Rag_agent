"""Document and retrieval domain models plus ingestion errors."""

from rag_agent.domain.documents import (
    SUFFIX_TO_FILE_TYPE,
    DocumentChunk,
    DocumentPage,
    DocumentStatus,
    FileType,
    SourceDocument,
    build_chunk_id,
    build_document_id,
    checksum_of,
    file_type_for_suffix,
    version_of,
)
from rag_agent.domain.errors import (
    ChunkingError,
    CorruptedDocumentError,
    DocumentNotFoundError,
    EmptyDocumentError,
    EncodingDetectionError,
    EncryptedDocumentError,
    IngestionError,
    MissingDependencyError,
    UnsupportedFileTypeError,
)
from rag_agent.domain.retrieval import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    RetrievalHit,
    RetrievalResult,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOP_K",
    "SUFFIX_TO_FILE_TYPE",
    "ChunkingError",
    "CorruptedDocumentError",
    "DocumentChunk",
    "DocumentNotFoundError",
    "DocumentPage",
    "DocumentStatus",
    "EmptyDocumentError",
    "EncodingDetectionError",
    "EncryptedDocumentError",
    "FileType",
    "IngestionError",
    "MissingDependencyError",
    "RetrievalHit",
    "RetrievalResult",
    "SourceDocument",
    "UnsupportedFileTypeError",
    "build_chunk_id",
    "build_document_id",
    "checksum_of",
    "file_type_for_suffix",
    "version_of",
]
