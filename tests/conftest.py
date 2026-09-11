"""Test-wide configuration.

The only job here is to place per-test temporary directories *inside the project*
instead of the system temp directory: ``conftest`` imports are loaded through the
``rootdir`` inserted at the front of ``sys.path``, so the support modules in
``tests/unit/`` can be imported by name from every test module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = PROJECT_ROOT / ".pytest_tmp"


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    """A clean per-test directory under ``<project>/.pytest_tmp``.

    This overrides pytest's built-in fixture of the same name. Creating
    ``mkdir(mode=0o700)`` directories under the system temp directory is blocked in
    the sandbox this project is developed in, and a path inside the repository
    works everywhere: each test gets its own directory, and ``.pytest_tmp/`` is
    ignored by Git so nothing leaks into a commit.
    """

    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in request.node.name
    )
    directory = TEST_TMP_ROOT / f"{safe_name}-{id(request.node):x}"
    if directory.exists():
        for child in sorted(directory.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
    directory.mkdir(parents=True, exist_ok=True)
    return directory
