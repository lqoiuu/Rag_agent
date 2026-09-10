"""Document domain models and ingestion errors."""

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
    CorruptedDocumentError,
    DocumentNotFoundError,
    EmptyDocumentError,
    EncodingDetectionError,
    EncryptedDocumentError,
    IngestionError,
    MissingDependencyError,
    UnsupportedFileTypeError,
)

__all__ = [
    "SUFFIX_TO_FILE_TYPE",
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
    "SourceDocument",
    "UnsupportedFileTypeError",
    "build_chunk_id",
    "build_document_id",
    "checksum_of",
    "file_type_for_suffix",
    "version_of",
]
