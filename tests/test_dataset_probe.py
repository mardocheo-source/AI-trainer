import json

from exotic_trainer.dataset_probe import inspect_dataset, normalize_record, tool_names_in_example


def test_expands_each_assistant_turn() -> None:
    examples = normalize_record(
        {
            "messages": [
                {"role": "user", "content": "inspect"},
                {"role": "assistant", "content": "read"},
                {"role": "tool", "content": "file"},
                {"role": "assistant", "content": "edit"},
            ]
        }
    )
    assert len(examples) == 2
    assert examples[1]["prompt"][-1]["role"] == "tool"


def test_local_probe_counts_jsonl_and_secrets(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    rows = [
        {"instruction": "hello", "output": "world"},
        {"instruction": "secret", "output": "ghp_" + "abcdefghijklmnopqrstuvwxyz" + "123456"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    probe = inspect_dataset(str(path))
    assert probe.row_count == 2
    assert probe.secret_hits == 1
    assert probe.sample_columns == ["instruction", "output"]


def test_converts_openai_tool_calls_to_lfm_and_pi_names() -> None:
    examples = normalize_record(
        {
            "messages": [
                {"role": "user", "content": "run tests"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "exec_command",
                                "arguments": '{"cmd":"pytest -q"}',
                            }
                        }
                    ],
                },
            ]
        }
    )
    content = examples[0]["completion"][0]["content"]
    assert content == "<|tool_call_start|>[bash(command='pytest -q')]<|tool_call_end|>"


def test_finds_unknown_native_tools() -> None:
    example = {
        "prompt": [{"role": "user", "content": "inspect"}],
        "completion": [
            {
                "role": "assistant",
                "content": "<|tool_call_start|>[lsp(operation='symbols')]<|tool_call_end|>",
            }
        ],
    }
    assert tool_names_in_example(example) == {"lsp"}


def test_converts_function_calling_reference_answer() -> None:
    examples = normalize_record(
        {
            "question": "Find a comedy movie",
            "reference_answer": "{'name': 'search_movies', 'arguments': {'genre': 'comedy'}}",
        }
    )

    assert examples[0]["completion"][0]["content"] == (
        "<|tool_call_start|>[search_movies(genre='comedy')]<|tool_call_end|>"
    )
