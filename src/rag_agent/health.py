"""Project health information used by CLI and tests."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rag_agent.config import PROJECT_ROOT

CORE_DISTRIBUTIONS = (
    "chromadb",
    "langchain",
    "langchain-chroma",
    "langgraph",
    "pydantic-settings",
)

CORE_MODULES = (
    "chromadb",
    "langchain",
    "langchain_chroma",
    "langgraph",
    "pydantic_settings",
)


@dataclass(frozen=True, slots=True)
class HealthReport:
    status: str
    checks: dict[str, bool]
    python_version: str
    executable: str
    base_prefix: str
    prefix: str
    expected_venv: str
    project_root: str
    dependencies: dict[str, str]
    imports: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in CORE_DISTRIBUTIONS:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _module_imports() -> dict[str, str]:
    imports: dict[str, str] = {}
    for module_name in CORE_MODULES:
        try:
            import_module(module_name)
        except Exception as exc:
            imports[module_name] = f"error: {type(exc).__name__}: {exc}"
        else:
            imports[module_name] = "ok"
    return imports


def collect_health() -> HealthReport:
    dependencies = _distribution_versions()
    imports = _module_imports()
    prefix = Path(sys.prefix).resolve()
    expected_venv = (PROJECT_ROOT / ".venv").resolve()
    checks = {
        "python_3_13": sys.version_info[:2] == (3, 13),
        "virtual_environment_active": sys.prefix != sys.base_prefix,
        "project_virtual_environment": prefix == expected_venv,
        "dependencies_installed": all(value != "not-installed" for value in dependencies.values()),
        "core_modules_importable": all(result == "ok" for result in imports.values()),
    }
    status = "ok" if all(checks.values()) else "degraded"

    return HealthReport(
        status=status,
        checks=checks,
        python_version=sys.version.split()[0],
        executable=str(Path(sys.executable).resolve()),
        base_prefix=str(Path(sys.base_prefix).resolve()),
        prefix=str(prefix),
        expected_venv=str(expected_venv),
        project_root=str(PROJECT_ROOT),
        dependencies=dependencies,
        imports=imports,
    )
