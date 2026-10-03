from exotic_trainer.agent_eval import (
    _expected_format_correct,
    _first_tool,
    _diagnostic_normalize_value,
    build_tool_menu,
    evaluation_slices_from_predictions,
    routing_metrics_from_predictions,
)
from exotic_trainer.dataset_probe import normalize_record
from exotic_trainer.server import parse_native_tool_calls
from exotic_trainer.token_roles import (
    TokenRole,
    python_role_spans,
    role_spans,
    roles_for_token_ids,
)


class CharacterTokenizer:
    def decode(self, token_ids, **_kwargs):
        return "".join(chr(value) for value in token_ids)


def test_extracts_native_lfm_tool_call() -> None:
    text = '<|tool_call_start|>[edit(path="x.py", oldText="a", newText="b")]<|tool_call_end|>'
    assert _first_tool(text) == "edit"


def test_plain_answer_has_no_tool() -> None:
    assert _first_tool("The constant is defined in config.py.") is None


def test_expected_tool_requires_balanced_parseable_native_call() -> None:
    prediction = "<|tool_call_start|>[read(path='x.py')]<|tool_call_end|>"
    _, calls = parse_native_tool_calls(prediction)

    assert _expected_format_correct(prediction, "read", "read", calls) is True
    assert _expected_format_correct("I would read x.py.", "read", None, []) is False


def test_no_tool_expected_requires_plain_response() -> None:
    assert _expected_format_correct("A direct answer.", None, None, []) is True
    prediction = "<|tool_call_start|>[read(path='x.py')]<|tool_call_end|>"
    _, calls = parse_native_tool_calls(prediction)
    assert _expected_format_correct(prediction, None, "read", calls) is False


def test_tool_menu_contains_required_schema_and_distractors() -> None:
    menu = build_tool_menu(["read", "bash", "lsp", "todowrite", "custom_tool"])
    functions = {item["function"]["name"]: item["function"] for item in menu}

    assert functions["read"]["parameters"]["required"] == ["path"]
    assert functions["bash"]["parameters"]["required"] == ["command"]
    assert functions["lsp"]["parameters"]["properties"]["line"]["type"] == "integer"
    assert functions["todowrite"]["parameters"]["properties"]["todos"]["type"] == "array"
    assert functions["custom_tool"]["parameters"]["additionalProperties"] is True


def test_routing_metrics_do_not_conflate_attempt_with_parseability() -> None:
    metrics = routing_metrics_from_predictions(
        [
            {"expected_tool": "write", "attempted_tool": True, "predicted_tool": None},
            {"expected_tool": None, "attempted_tool": True, "predicted_tool": None},
        ]
    )

    assert metrics["tool_recall"] == 1.0
    assert metrics["no_tool_specificity"] == 0.0
    assert metrics["balanced_accuracy"] == 0.5


def test_dev_metadata_is_reported_as_separate_condition_slices() -> None:
    predictions = [
        {
            "evaluation_metadata": {"condition": "neutral", "stratum": "clean"},
            "expected_tool": None,
            "attempted_tool": True,
            "expected_format_correct": False,
        },
        {
            "evaluation_metadata": {"condition": "policy", "stratum": "clean"},
            "expected_tool": None,
            "attempted_tool": False,
            "expected_format_correct": True,
        },
    ]

    slices = evaluation_slices_from_predictions(predictions)

    assert slices["condition"]["neutral"]["no_tool_specificity"] == 0.0
    assert slices["condition"]["policy"]["no_tool_specificity"] == 1.0


def test_lsp_file_path_is_not_rewritten_to_generic_path() -> None:
    normalized = normalize_record(
        {
            "messages": [
                {"role": "user", "content": "Find the definition."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lsp",
                                "arguments": {
                                    "operation": "goToDefinition",
                                    "filePath": "src/app.py",
                                    "line": 4,
                                    "character": 2,
                                },
                            }
                        }
                    ],
                },
            ]
        }
    )

    completion = normalized[0]["completion"][0]["content"]
    assert "filePath='src/app.py'" in completion
    assert "path='src/app.py'" not in completion


def test_native_call_tokens_are_partitioned_into_structural_roles() -> None:
    text = "<|tool_call_start|>[read(path='x')]<|tool_call_end|>"
    spans = role_spans(text)
    extracted = {(text[start:end], role) for start, end, role in spans}
    roles = roles_for_token_ids(CharacterTokenizer(), [ord(char) for char in text])

    assert ("read", TokenRole.TOOL_NAME) in extracted
    assert ("path", TokenRole.ARGUMENT_KEY) in extracted
    assert ("'x'", TokenRole.ARGUMENT_VALUE) in extracted
    assert roles[text.index("read")] == TokenRole.TOOL_NAME


def test_python_state_tokens_include_producer_and_consumer_roles() -> None:
    text = "users = search_users(city='Kobe')\nsent = notify(recipients=users)"
    extracted = {(text[start:end], role) for start, end, role in python_role_spans(text)}

    assert ("search_users", TokenRole.TOOL_NAME) in extracted
    assert ("recipients", TokenRole.ARGUMENT_KEY) in extracted
    assert ("users", TokenRole.ARGUMENT_VALUE) in extracted
    assert sum(value == ("users", TokenRole.ARGUMENT_VALUE) for value in extracted) >= 1


def test_direct_text_does_not_create_python_tool_roles() -> None:
    assert python_role_spans("I need the city before I can continue.") == []


def test_diagnostic_normalization_only_relaxes_terminal_newlines_for_payloads() -> None:
    assert _diagnostic_normalize_value("content", "a\r\n") == "a"
    assert _diagnostic_normalize_value("path", "a\n") == "a\n"
