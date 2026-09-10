from pathlib import Path

from rag_agent.config.settings import PROJECT_ROOT, Settings


def test_relative_paths_resolve_from_project_root() -> None:
    settings = Settings(
        _env_file=None,
        data_dir=Path("data/test"),
        chroma_dir=Path("data/test/chroma"),
        sqlite_path=Path("data/test/rag_agent.sqlite3"),
    )

    assert settings.data_dir == (PROJECT_ROOT / "data/test").resolve()
    assert settings.chroma_dir == (PROJECT_ROOT / "data/test/chroma").resolve()
    assert settings.sqlite_path == (PROJECT_ROOT / "data/test/rag_agent.sqlite3").resolve()


def test_api_key_is_optional_during_stage_one() -> None:
    settings = Settings(_env_file=None, qwen_api_key=None)

    assert settings.qwen_api_key is None
