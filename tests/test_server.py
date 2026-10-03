import json

from exotic_trainer.server import (
    _sanitize_json_value,
    _use_kv_cache,
    parse_native_tool_calls,
)


def test_parses_lfm_native_call_for_openai_api() -> None:
    content, calls = parse_native_tool_calls(
        'Checking.<|tool_call_start|>[bash(command="pytest -q", timeout=120)]'
        "<|tool_call_end|>"
    )
    assert content == "Checking."
    assert calls[0]["function"]["name"] == "bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "command": "pytest -q",
        "timeout": 120,
    }


def test_malformed_call_stays_as_text() -> None:
    text = "<|tool_call_start|>[bash(command=unknown + 1)]<|tool_call_end|>"
    content, calls = parse_native_tool_calls(text)
    assert content == text
    assert calls == []


def test_json_sanitizer_removes_surrogates_recursively() -> None:
    assert _sanitize_json_value({"value": "a\ud800b", "items": ["\udfff"]}) == {
        "value": "ab",
        "items": [""],
    }


def test_long_prompts_disable_kv_cache() -> None:
    assert _use_kv_cache(2_048, 2_048) is True
    assert _use_kv_cache(2_049, 2_048) is False
