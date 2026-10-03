import json
from types import SimpleNamespace

import pytest

from exotic_trainer import bfcl
from exotic_trainer.server import parse_generic_tool_calls


def test_bfcl_v3_full_uses_historical_all_collection(monkeypatch) -> None:
    monkeypatch.setattr(
        bfcl,
        "_resolve_target",
        lambda _target: ("/models/lfm", None, "lfm"),
    )
    monkeypatch.setattr(
        bfcl.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=20 * bfcl.GIB),
    )

    config = bfcl.build_bfcl_config(
        version="v3",
        profile="full",
        handler="liquid-lfm2",
        transport="managed-local-xpu",
        target="model:/models/lfm",
        endpoint="",
        api_model="local-model",
        smoke_samples=20,
    )

    assert config["categories"] == ["all"]
    assert config["bfcl_commit"] == bfcl.BFCL_SPECS["v3"]["commit"]
    assert config["runtime_limits"] == {
        "request_timeout_seconds": 210.0,
        "generation_timeout_seconds": 180.0,
        "openai_max_retries": 0,
        "long_prompt_cache_threshold": 4096,
        "xpu_memory_fraction": 0.85,
        "max_new_tokens": 1024,
    }


def test_smoke_is_partial_and_v3_rejects_it(monkeypatch) -> None:
    monkeypatch.setattr(
        bfcl,
        "_resolve_target",
        lambda _target: ("/models/lfm", None, "lfm"),
    )
    config = bfcl.build_bfcl_config(
        version="v4",
        profile="smoke",
        handler="generic-openai-tools",
        transport="managed-local-xpu",
        target="model:/models/lfm",
        endpoint="",
        api_model="local-model",
        smoke_samples=12,
    )
    assert config["partial"] is True
    assert config["smoke_samples"] == 12

    with pytest.raises(ValueError, match="does not support honest partial scoring"):
        bfcl.build_bfcl_config(
            version="v3",
            profile="smoke",
            handler="liquid-lfm2",
            transport="managed-local-xpu",
            target="model:/models/lfm",
            endpoint="",
            api_model="local-model",
            smoke_samples=12,
        )


def test_v4_full_is_refused_in_compact_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        bfcl,
        "_resolve_target",
        lambda _target: ("/models/lfm", None, "lfm"),
    )
    with pytest.raises(ValueError, match="heavy agentic"):
        bfcl.build_bfcl_config(
            version="v4",
            profile="full",
            handler="liquid-lfm2",
            transport="managed-local-xpu",
            target="model:/models/lfm",
            endpoint="",
            api_model="local-model",
            smoke_samples=12,
        )


@pytest.mark.parametrize(
    "response",
    [
        '<tool_call>{"name":"read","arguments":{"path":"a.py"}}</tool_call>',
        '{"function":{"name":"read","arguments":"{\\"path\\":\\"a.py\\"}"}}',
        '{"tool_calls":[{"function":{"name":"read","arguments":{"path":"a.py"}}}]}',
        'read(path="a.py")',
    ],
)
def test_generic_gateway_parses_common_tool_call_formats(response: str) -> None:
    content, calls = parse_generic_tool_calls(response)

    assert content == ""
    assert calls[0]["function"]["name"] == "read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}


def test_generic_gateway_does_not_treat_json_in_prose_as_tool_call() -> None:
    content, calls = parse_generic_tool_calls('Here is ordinary JSON: {"x": 1}')

    assert content == 'Here is ordinary JSON: {"x": 1}'
    assert calls == []


def test_liquid_parser_supports_dotted_names_barewords_and_thinking() -> None:
    response = (
        "<think>private reasoning</think>"
        "<|tool_call_start|>[repo.read(path=src_main, lines=[1, -2])]<|tool_call_end|>"
    )
    content, calls = parse_generic_tool_calls(response)

    assert content == ""
    assert calls[0]["function"]["name"] == "repo.read"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "path": "src_main",
        "lines": [1, -2],
    }


def test_seed_results_keeps_good_rows_and_retries_errors(tmp_path) -> None:
    source = tmp_path / "source"
    source_result = source / "result" / "model"
    source_result.mkdir(parents=True)
    (source_result / "cases.json").write_text(
        '\n'.join(
            [
                json.dumps({"id": "good", "result": [{"read": "{}"}]}),
                json.dumps({"id": "retry", "result": "Error during inference: timed out"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    config_path = output / "config.json"
    config_path.write_text(json.dumps({"output_dir": str(output)}), encoding="utf-8")

    recovery = bfcl.seed_bfcl_results(config_path, source)

    rows = [
        json.loads(line)
        for line in (output / "result" / "model" / "cases.json").read_text().splitlines()
    ]
    assert recovery["retained"] == 1
    assert recovery["retry"] == 1
    assert [row["id"] for row in rows] == ["good"]


def test_score_only_reuses_complete_predictions_without_gateway(tmp_path, monkeypatch) -> None:
    output = tmp_path / "bfcl-run"
    result = output / "result"
    result.mkdir(parents=True)
    (result / "BFCL_v3_simple_result.json").write_text(
        '{"id":"a","result":[]}\n{"id":"b","result":[]}\n',
        encoding="utf-8",
    )
    config_path = output / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "version": "v3",
                "output_dir": str(output),
                "partial": False,
            }
        ),
        encoding="utf-8",
    )
    (output / "state.json").write_text(
        json.dumps({"status": "failed", "total_cases": 2}), encoding="utf-8"
    )
    monkeypatch.setattr(bfcl, "_benchmark_dir", lambda _version: tmp_path / "source")

    seen: list[str] = []

    def fake_run(command, **_kwargs):
        seen.extend(str(item) for item in command)
        (output / "official-result.json").write_text(
            json.dumps({"complete": True, "score_only": True}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bfcl.subprocess, "run", fake_run)

    state = bfcl.score_existing_bfcl(config_path)

    assert state["status"] == "complete"
    assert state["score_recovery"]["predictions_reused"] == 2
    assert "--score-only" in seen
    assert "serve" not in seen
