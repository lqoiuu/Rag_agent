"""SQLite metadata storage for documents, versions and ingestion jobs.

The store is a thin typed wrapper over ``sqlite3``: it owns the schema and
returns domain records, so the pipeline never sees raw rows.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        document_id TEXT PRIMARY KEY,
        source TEXT NOT NULL UNIQUE,
        file_type TEXT NOT NULL,
        checksum TEXT NOT NULL,
        version TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        page_count INTEGER NOT NULL,
        chunk_count INTEGER NOT NULL,
        embedding_model TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_versions (
        document_id TEXT NOT NULL,
        version TEXT NOT NULL,
        checksum TEXT NOT NULL,
        chunk_count INTEGER NOT NULL,
        embedding_model TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (document_id, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_jobs (
        job_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        document_id TEXT,
        status TEXT NOT NULL,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        error_code TEXT,
        error_message TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_source ON ingestion_jobs (source, finished_at)",
)


def utc_now() -> str:
    """Return a second-precision UTC timestamp for records."""

    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class StoredDocument:
    """Current indexed state of one source document."""

    document_id: str
    source: str
    file_type: str
    checksum: str
    version: str
    size_bytes: int
    page_count: int
    chunk_count: int
    embedding_model: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    """One content version seen for a document."""

    document_id: str
    version: str
    checksum: str
    chunk_count: int
    embedding_model: str
    created_at: str


@dataclass(frozen=True, slots=True)
class IngestionJob:
    """One ingestion attempt, successful or not."""

    job_id: str
    source: str
    status: str
    started_at: str
    finished_at: str
    document_id: str | None = None
    chunk_count: int = 0
    error_code: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "source": self.source,
            "document_id": self.document_id,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class MetadataStore:
    """Typed access to the project metadata database.

    ``":memory:"`` is accepted as a path so tests can run without a file.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._connection = sqlite3.connect(self._path)
        self._connection.row_factory = sqlite3.Row
        self.initialize()

    @property
    def path(self) -> str:
        return self._path

    def initialize(self) -> None:
        """Create the schema when it is missing; safe to call repeatedly."""

        with self._connection:
            for statement in SCHEMA:
                self._connection.execute(statement)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> MetadataStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get_document(self, source: str) -> StoredDocument | None:
        row = self._connection.execute(
            "SELECT * FROM documents WHERE source = ?", (source,)
        ).fetchone()
        return None if row is None else _document_from_row(row)

    def list_documents(self) -> tuple[StoredDocument, ...]:
        rows = self._connection.execute("SELECT * FROM documents ORDER BY source").fetchall()
        return tuple(_document_from_row(row) for row in rows)

    def save_document(self, document: StoredDocument) -> None:
        """Insert or replace the current state and record the version."""

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO documents (
                    document_id, source, file_type, checksum, version,
                    size_bytes, page_count, chunk_count, embedding_model, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    document_id = excluded.document_id,
                    file_type = excluded.file_type,
                    checksum = excluded.checksum,
                    version = excluded.version,
                    size_bytes = excluded.size_bytes,
                    page_count = excluded.page_count,
                    chunk_count = excluded.chunk_count,
                    embedding_model = excluded.embedding_model,
                    updated_at = excluded.updated_at
                """,
                (
                    document.document_id,
                    document.source,
                    document.file_type,
                    document.checksum,
                    document.version,
                    document.size_bytes,
                    document.page_count,
                    document.chunk_count,
                    document.embedding_model,
                    document.updated_at,
                ),
            )
            self._connection.execute(
                """
                INSERT OR REPLACE INTO document_versions (
                    document_id, version, checksum, chunk_count, embedding_model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document.document_id,
                    document.version,
                    document.checksum,
                    document.chunk_count,
                    document.embedding_model,
                    document.updated_at,
                ),
            )

    def delete_document(self, source: str) -> StoredDocument | None:
        """Remove a document and its version history; returns what was removed."""

        existing = self.get_document(source)
        if existing is None:
            return None
        with self._connection:
            self._connection.execute("DELETE FROM documents WHERE source = ?", (source,))
            self._connection.execute(
                "DELETE FROM document_versions WHERE document_id = ?", (existing.document_id,)
            )
        return existing

    def list_versions(self, document_id: str) -> tuple[DocumentVersion, ...]:
        rows = self._connection.execute(
            """
            SELECT * FROM document_versions
            WHERE document_id = ?
            ORDER BY created_at, version
            """,
            (document_id,),
        ).fetchall()
        return tuple(_version_from_row(row) for row in rows)

    def record_job(self, job: IngestionJob) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO ingestion_jobs (
                    job_id, source, document_id, status, chunk_count,
                    error_code, error_message, started_at, finished_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.job_id,
                    job.source,
                    job.document_id,
                    job.status,
                    job.chunk_count,
                    job.error_code,
                    job.error_message,
                    job.started_at,
                    job.finished_at,
                ),
            )

    def latest_job(self, source: str) -> IngestionJob | None:
        row = self._connection.execute(
            """
            SELECT * FROM ingestion_jobs
            WHERE source = ?
            ORDER BY finished_at DESC, rowid DESC
            LIMIT 1
            """,
            (source,),
        ).fetchone()
        return None if row is None else _job_from_row(row)

    def list_jobs(self, limit: int = 20) -> tuple[IngestionJob, ...]:
        rows = self._connection.execute(
            """
            SELECT * FROM ingestion_jobs
            ORDER BY finished_at DESC, rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return tuple(_job_from_row(row) for row in rows)


def _document_from_row(row: sqlite3.Row) -> StoredDocument:
    return StoredDocument(
        document_id=str(row["document_id"]),
        source=str(row["source"]),
        file_type=str(row["file_type"]),
        checksum=str(row["checksum"]),
        version=str(row["version"]),
        size_bytes=int(row["size_bytes"]),
        page_count=int(row["page_count"]),
        chunk_count=int(row["chunk_count"]),
        embedding_model=str(row["embedding_model"]),
        updated_at=str(row["updated_at"]),
    )


def _version_from_row(row: sqlite3.Row) -> DocumentVersion:
    return DocumentVersion(
        document_id=str(row["document_id"]),
        version=str(row["version"]),
        checksum=str(row["checksum"]),
        chunk_count=int(row["chunk_count"]),
        embedding_model=str(row["embedding_model"]),
        created_at=str(row["created_at"]),
    )


def _job_from_row(row: sqlite3.Row) -> IngestionJob:
    document_id = row["document_id"]
    error_code = row["error_code"]
    error_message = row["error_message"]
    return IngestionJob(
        job_id=str(row["job_id"]),
        source=str(row["source"]),
        status=str(row["status"]),
        started_at=str(row["started_at"]),
        finished_at=str(row["finished_at"]),
        document_id=None if document_id is None else str(document_id),
        chunk_count=int(row["chunk_count"]),
        error_code=None if error_code is None else str(error_code),
        error_message=None if error_message is None else str(error_message),
    )
