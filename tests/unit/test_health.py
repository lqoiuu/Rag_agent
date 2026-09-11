from pathlib import Path

import pytest

from rag_agent.config import PROJECT_ROOT
from rag_agent.health import (
    CORE_DISTRIBUTIONS,
    CORE_MODULES,
    _expected_virtual_environment,
    collect_health,
)


def test_health_reports_python_and_core_dependencies() -> None:
    report = collect_health()

    assert report.status == "ok"
    assert all(report.checks.values())
    assert report.python_version.startswith("3.13.")
    assert Path(report.prefix) == (PROJECT_ROOT / ".venv").resolve()
    assert set(report.dependencies) == set(CORE_DISTRIBUTIONS)
    assert all(version != "not-installed" for version in report.dependencies.values())
    assert set(report.imports) == set(CORE_MODULES)
    assert all(result == "ok" for result in report.imports.values())


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (".container-venv", PROJECT_ROOT / ".container-venv"),
        (str(PROJECT_ROOT / "absolute-venv"), PROJECT_ROOT / "absolute-venv"),
    ],
)
def test_expected_virtual_environment_honors_uv_configuration(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: Path,
) -> None:
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", configured)

    assert _expected_virtual_environment() == expected.resolve()
