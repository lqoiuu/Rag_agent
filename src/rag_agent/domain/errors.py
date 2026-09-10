"""Domain errors for document ingestion.

Every error carries a stable ``code`` so callers, logs and future API responses
never have to match on message text.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for failures while loading or normalizing a document."""

    code: str = "ingestion_error"

    def __init__(self, message: str, *, source: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.source = source


class DocumentNotFoundError(IngestionError):
    """The requested file or directory does not exist."""

    code = "document_not_found"


class UnsupportedFileTypeError(IngestionError):
    """The file suffix has no registered loader."""

    code = "unsupported_file_type"


class EmptyDocumentError(IngestionError):
    """The file exists but yields no visible text."""

    code = "empty_document"


class EncodingDetectionError(IngestionError):
    """No candidate encoding could decode the raw bytes."""

    code = "encoding_error"


class EncryptedDocumentError(IngestionError):
    """The PDF needs a password the project does not store."""

    code = "encrypted_document"


class CorruptedDocumentError(IngestionError):
    """The file content cannot be parsed by its loader."""

    code = "corrupted_document"


class MissingDependencyError(IngestionError):
    """An optional parser dependency is not installed."""

    code = "missing_dependency"
