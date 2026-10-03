"""Credential controls use only explicit synthetic test values."""

import importlib.util

import pytest

KEYS = ("RAPID_API_KEY", "GEOCODE_API_KEY", "EXCHANGERATE_API_KEY", "OMDB_API_KEY")


def loader():
    assert importlib.util.find_spec("exotic_trainer.credentials") is not None, (
        "credential loader missing"
    )
    from exotic_trainer.credentials import load_bfcl_environment

    return load_bfcl_environment


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for name in (*KEYS, "AI_TRAINER_ENV_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("EXOTIC_TRAINER_ROOT", str(tmp_path))


def test_loads_local_file_and_preserves_shell(monkeypatch, tmp_path):
    import os

    directory = tmp_path / ".secrets"
    directory.mkdir()
    (directory / "bfcl.env").write_text(
        "RAPID_API_KEY=synthetic-file-value\nOMDB_API_KEY=synthetic-omdb\nUNRELATED=ignored\n"
    )
    monkeypatch.setenv("RAPID_API_KEY", "synthetic-shell-value")
    loader()()
    assert os.environ["RAPID_API_KEY"] == "synthetic-shell-value"
    assert os.environ["OMDB_API_KEY"] == "synthetic-omdb"
    assert "UNRELATED" not in os.environ


def test_explicit_file_supports_spaces_and_literal_dollars(monkeypatch, tmp_path):
    import os

    path = tmp_path / "my local.env"
    path.write_text('GEOCODE_API_KEY="synthetic-${HOME}-value"\n')
    monkeypatch.setenv("AI_TRAINER_ENV_FILE", str(path))
    loader()()
    assert os.environ["GEOCODE_API_KEY"] == "synthetic-${HOME}-value"


def test_offline_has_no_fake_credentials():
    import os

    loader()()
    assert all(os.environ[name] == "" for name in KEYS)


def test_live_missing_keys_reports_names_only(monkeypatch):
    sentinel = "synthetic-sensitive-sentinel"
    monkeypatch.setenv("RAPID_API_KEY", sentinel)
    with pytest.raises(ValueError) as error:
        loader()(require_live=True)
    assert "OMDB_API_KEY" in str(error.value)
    assert sentinel not in str(error.value)


def test_live_rejects_placeholders(monkeypatch):
    for name in KEYS:
        monkeypatch.setenv(name, "placeholder-key-for-local-eval")
    with pytest.raises(ValueError, match="RAPID_API_KEY"):
        loader()(require_live=True)


def test_explicit_missing_file_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_TRAINER_ENV_FILE", str(tmp_path / "absent.env"))
    with pytest.raises(FileNotFoundError, match="AI_TRAINER_ENV_FILE"):
        loader()()


def test_relative_file_remains_valid_after_runner_changes_directory(monkeypatch, tmp_path):
    path = tmp_path / "live.env"
    path.write_text("".join(f"{name}=synthetic-live-value\n" for name in KEYS))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_TRAINER_ENV_FILE", "live.env")
    load = loader()
    load()
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.chdir(output)
    load(require_live=True)
