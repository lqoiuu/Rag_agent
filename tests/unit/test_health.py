from pathlib import Path

from rag_agent.config import PROJECT_ROOT
from rag_agent.health import CORE_DISTRIBUTIONS, CORE_MODULES, collect_health


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
