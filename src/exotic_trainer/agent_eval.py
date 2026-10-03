from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
import shlex
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .tool_schema import ARGUMENT_TYPES, KNOWN_TOOLS, REQUIRED_ARGUMENTS, build_tool_menu

TOOL_PATTERN = re.compile(r"\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _diagnostic_normalize_value(key: str, value: Any) -> Any:
    """Conservative diagnostic normalization, never used as the strict score."""
    if isinstance(value, str):
        normalized = value.replace("\r\n", "\n")
        if key in {"content", "patchText"}:
            return normalized.rstrip("\n")
        return normalized
    if isinstance(value, list):
        return [_diagnostic_normalize_value(key, item) for item in value]
    if isinstance(value, dict):
        return {
            child_key: _diagnostic_normalize_value(str(child_key), child)
            for child_key, child in value.items()
        }
    return value


def _elastic_normalize_value(key: str, value: Any, expected: Any) -> Any:
    """Normalize only representation differences that preserve tool semantics.

    This score deliberately remains conservative.  It accepts JSON scalar
    spellings (``"30"`` vs ``30``), CRLF/newline presentation, redundant path
    segments and shell quoting that produces the same argv.  It never changes a
    literal's words, silently drops a required field or repairs a wrong tool.
    Strict byte/value metrics are reported alongside it.
    """

    if isinstance(expected, bool):
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        return value
    if isinstance(expected, int) and not isinstance(expected, bool):
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value.strip())
        return value
    if isinstance(expected, float):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return value
    if isinstance(expected, str) and isinstance(value, str):
        normalized = value.replace("\r\n", "\n")
        if key in {"content", "patchText"}:
            return normalized.rstrip("\n")
        if key in {"path", "filePath"}:
            return posixpath.normpath(normalized)
        if key == "command":
            try:
                return ("shell-argv", tuple(shlex.split(normalized)))
            except ValueError:
                return normalized
        # Preserve exact semantics for regexes, queries, symbols and prose.
        return normalized
    if isinstance(expected, list) and isinstance(value, list):
        return [
            _elastic_normalize_value(key, item, expected[index] if index < len(expected) else item)
            for index, item in enumerate(value)
        ]
    if isinstance(expected, dict) and isinstance(value, dict):
        return {
            child_key: _elastic_normalize_value(
                str(child_key), child, expected.get(child_key, child)
            )
            for child_key, child in value.items()
        }
    return value


def _elastic_values_equal(key: str, predicted: Any, expected: Any) -> bool:
    return _elastic_normalize_value(key, predicted, expected) == _elastic_normalize_value(
        key, expected, expected
    )


def _content(completion: Any) -> str:
    if isinstance(completion, list) and completion:
        value = completion[0]
        if isinstance(value, dict):
            return str(value.get("content", ""))
    return str(completion)


def _first_tool(text: str) -> str | None:
    match = TOOL_PATTERN.search(text)
    return match.group(1) if match else None


def _expected_format_correct(
    prediction: str,
    expected_tool: str | None,
    predicted_tool: str | None,
    predicted_calls: list[dict[str, Any]],
) -> bool:
    """Check whether the response shape matches the expected tool/no-tool decision."""
    has_open = "<|tool_call_start|>" in prediction
    has_close = "<|tool_call_end|>" in prediction
    if expected_tool is not None:
        return has_open and has_close and bool(predicted_calls)
    return not has_open and not has_close and predicted_tool is None and not predicted_calls


def _prompt_has_embedded_tool_menu(prompt: Any) -> bool:
    return bool(
        isinstance(prompt, list)
        and prompt
        and isinstance(prompt[0], dict)
        and prompt[0].get("role") == "system"
        and "List of tools: [" in str(prompt[0].get("content") or "")
    )


def routing_metrics_from_predictions(
    predictions: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Recompute decision-only metrics from stored per-example predictions.

    Schema 3/4 result files recorded enough evidence to repair their legacy
    parseability/decision conflation without rerunning expensive inference.
    """
    tool_rows = [item for item in predictions if item.get("expected_tool") is not None]
    no_tool_rows = [item for item in predictions if item.get("expected_tool") is None]
    true_positive = sum(bool(item.get("attempted_tool")) for item in tool_rows)
    true_negative = sum(not bool(item.get("attempted_tool")) for item in no_tool_rows)
    false_negative = len(tool_rows) - true_positive
    false_positive = len(no_tool_rows) - true_negative
    recall = true_positive / max(1, len(tool_rows))
    specificity = true_negative / max(1, len(no_tool_rows))
    product = (
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        (true_positive * true_negative - false_positive * false_negative) / math.sqrt(product)
        if product
        else 0.0
    )
    return {
        "tool_required_samples": len(tool_rows),
        "no_tool_samples": len(no_tool_rows),
        "tool_true_positive": true_positive,
        "no_tool_true_negative": true_negative,
        "tool_recall": recall,
        "no_tool_specificity": specificity,
        "accuracy": (true_positive + true_negative) / max(1, len(predictions)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "mcc": mcc,
    }


def evaluation_slices_from_predictions(
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Summarize declared DEV metadata without using it as a model input."""
    output: dict[str, dict[str, dict[str, float | int]]] = {}
    for dimension in ("condition", "stratum"):
        values = sorted(
            {
                str((item.get("evaluation_metadata") or {}).get(dimension))
                for item in predictions
                if (item.get("evaluation_metadata") or {}).get(dimension) is not None
            }
        )
        if not values:
            continue
        dimension_result: dict[str, dict[str, float | int]] = {}
        for value in values:
            selected = [
                item
                for item in predictions
                if str((item.get("evaluation_metadata") or {}).get(dimension)) == value
            ]
            routing = routing_metrics_from_predictions(selected)
            tool_rows = [item for item in selected if item.get("expected_tool") is not None]
            dimension_result[value] = {
                "samples": len(selected),
                "tool_required_samples": len(tool_rows),
                "expected_format_accuracy": sum(
                    bool(item.get("expected_format_correct")) for item in selected
                )
                / max(1, len(selected)),
                "balanced_routing_accuracy": float(routing["balanced_accuracy"]),
                "tool_recall": float(routing["tool_recall"]),
                "no_tool_specificity": float(routing["no_tool_specificity"]),
                "routing_mcc": float(routing["mcc"]),
                "selected_tool_name_accuracy": sum(
                    bool(item.get("selected_tool_name_correct")) for item in tool_rows
                )
                / max(1, len(tool_rows)),
                "parseable_call_rate": sum(bool(item.get("predicted_tool")) for item in tool_rows)
                / max(1, len(tool_rows)),
                "exact_arguments_accuracy": sum(
                    bool(item.get("argument_values_exact")) for item in tool_rows
                )
                / max(1, len(tool_rows)),
                "required_argument_accuracy": sum(
                    bool(item.get("required_argument_values_exact")) for item in tool_rows
                )
                / max(1, len(tool_rows)),
                "elastic_required_argument_accuracy": sum(
                    bool(item.get("required_argument_values_elastic")) for item in tool_rows
                )
                / max(1, len(tool_rows)),
                "argument_field_micro_accuracy": sum(
                    int(item.get("argument_fields_exact") or 0) for item in tool_rows
                )
                / max(
                    1,
                    sum(int(item.get("argument_fields_expected") or 0) for item in tool_rows),
                ),
            }
        output[dimension] = dimension_result
    return output


def evaluate_agent_behavior(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_samples: int,
    max_new_tokens: int,
    deadline: float,
    stop_file: Path | None = None,
    tool_names: list[str] | None = None,
    include_predictions: bool = False,
) -> dict[str, Any]:
    if max_samples <= 0:
        return {"agent_eval_samples": 0}
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    old_cache = getattr(model.config, "use_cache", False)
    model.eval()
    model.config.use_cache = True
    totals = {
        "agent_metric_schema_version": 8,
        "agent_execution_evaluated": False,
        "agent_eval_samples": 0,
        "agent_format_valid": 0,
        "agent_expected_format_correct": 0,
        "agent_tool_decision_correct": 0,
        "agent_regex_tool_decision_correct": 0,
        "agent_tool_required_samples": 0,
        "agent_no_tool_samples": 0,
        "agent_tool_true_positive": 0,
        "agent_parseable_tool_true_positive": 0,
        "agent_parseable_tool_decision_correct": 0,
        "agent_no_tool_true_negative": 0,
        "agent_tool_name_correct": 0,
        "agent_selected_tool_name_correct": 0,
        "agent_tool_name_expected": 0,
        "agent_tool_name_format_eligible": 0,
        "agent_tool_name_correct_given_expected_format": 0,
        "agent_tool_arguments_valid": 0,
        "agent_tool_argument_keys_correct": 0,
        "agent_tool_argument_values_exact": 0,
        "agent_required_argument_calls_exact": 0,
        "agent_required_argument_calls_normalized": 0,
        "agent_required_argument_calls_elastic": 0,
        "agent_argument_fields_expected": 0,
        "agent_argument_fields_exact": 0,
        "agent_required_argument_fields_expected": 0,
        "agent_required_argument_fields_exact": 0,
        "agent_tool_arguments_typed": 0,
        "agent_exact_call_count": 0,
        "agent_nonempty_no_tool_answer": 0,
        "agent_no_tool_answer_exact": 0,
        "agent_direct_criteria_samples": 0,
        "agent_direct_required_terms_pass": 0,
        "agent_human_task_success": 0,
        "agent_strict_human_task_success": 0,
        "agent_elastic_human_task_success": 0,
    }
    predictions: list[dict[str, Any]] = []
    per_tool_expected: Counter[str] = Counter()
    per_tool_correct: Counter[str] = Counter()
    per_tool_whole_exact: Counter[str] = Counter()
    per_tool_required_exact: Counter[str] = Counter()
    per_tool_required_elastic: Counter[str] = Counter()
    per_tool_fields_expected: Counter[str] = Counter()
    per_tool_fields_exact: Counter[str] = Counter()
    per_tool_required_fields_expected: Counter[str] = Counter()
    per_tool_required_fields_exact: Counter[str] = Counter()
    schema_menu = build_tool_menu(tool_names or sorted(KNOWN_TOOLS))
    canonical_menu = json.dumps(
        schema_menu, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    protocol_payload = {
        "metric_schema": totals["agent_metric_schema_version"],
        "parser": "lfm-native-ast-v1",
        "tool_menu": json.loads(canonical_menu),
        "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
    }
    totals["agent_tool_menu_sha256"] = hashlib.sha256(canonical_menu.encode()).hexdigest()
    totals["agent_chat_template_sha256"] = protocol_payload["chat_template_sha256"]
    totals["agent_protocol_id"] = hashlib.sha256(
        json.dumps(
            protocol_payload,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    totals["agent_generation_config"] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    schema_applied = True
    try:
        from .server import parse_native_tool_calls

        for index in range(min(max_samples, len(dataset))):
            if (index + 1) % 25 == 0 or index == 0 or (index + 1) == min(max_samples, len(dataset)):
                print(f"  • Eval progress: [{index+1}/{min(max_samples, len(dataset))}] ({((index+1)/min(max_samples, len(dataset)))*100:.1f}%)", flush=True)
            if time.monotonic() >= deadline - 10:
                break
            if stop_file is not None and stop_file.exists():
                break
            row = dataset[index]
            prompt = row["prompt"]
            expected = _content(row["completion"])
            try:
                evaluation_metadata = json.loads(str(row.get("evaluation_metadata_json") or "{}"))
            except json.JSONDecodeError:
                evaluation_metadata = {}
            template_kwargs: dict[str, Any] = {
                "conversation": prompt,
                "add_generation_prompt": True,
                "tokenize": True,
                "return_tensors": "pt",
                "return_dict": True,
            }
            # Training-time menu conditioning embeds the exact LFM tool text in
            # the system message. Passing ``tools`` again would duplicate the
            # full contract at evaluation time and make train/DEV protocols
            # incomparable.
            if not _prompt_has_embedded_tool_menu(prompt):
                template_kwargs["tools"] = schema_menu
            try:
                encoded = tokenizer.apply_chat_template(**template_kwargs).to(device)
            except (TypeError, ValueError):
                # Older/custom tokenizers may not expose a tools argument. The
                # result explicitly records the fallback so protocols cannot
                # silently mix schema-aware and schema-free evaluation.
                schema_applied = False
                template_kwargs.pop("tools", None)
                encoded = tokenizer.apply_chat_template(**template_kwargs).to(device)
            input_ids = encoded["input_ids"]
            prompt_len = input_ids.shape[-1]
            penalty = float(os.environ.get("EXOTIC_TRAINER_TOOL_START_PENALTY", "0.0"))
            processors = None
            if penalty > 0.0:
                from transformers import LogitsProcessor, LogitsProcessorList

                class FirstTokenGating(LogitsProcessor):
                    def __init__(self, p_len: int, pen: float):
                        self.p_len = p_len
                        self.pen = pen
                    def __call__(self, i_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                        if i_ids.shape[1] == self.p_len:
                            # Token ID 10 is <|tool_call_start|>
                            scores[:, 10] -= self.pen
                        return scores

                processors = LogitsProcessorList([FirstTokenGating(prompt_len, penalty)])

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    logits_processor=processors,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prediction = tokenizer.decode(
                generated[0, input_ids.shape[-1] :], skip_special_tokens=False
            )
            regex_predicted_tool = _first_tool(prediction)
            expected_tool = _first_tool(expected)
            predicted_content, predicted_calls = parse_native_tool_calls(prediction)
            _, expected_calls = parse_native_tool_calls(expected)
            predicted_tool = (
                str(predicted_calls[0]["function"]["name"]) if predicted_calls else None
            )
            has_open = "<|tool_call_start|>" in prediction
            has_close = "<|tool_call_end|>" in prediction
            attempted_tool = (
                bool(predicted_calls) or has_open or has_close or bool(regex_predicted_tool)
            )
            sample_known_tools = set(tool_names or KNOWN_TOOLS) | (
                {expected_tool} if expected_tool else set()
            )
            valid = has_open == has_close and (
                regex_predicted_tool is None or regex_predicted_tool in sample_known_tools
            )
            expected_format_correct = _expected_format_correct(
                prediction, expected_tool, regex_predicted_tool, predicted_calls
            )
            totals["agent_eval_samples"] += 1
            totals["agent_format_valid"] += int(valid)
            totals["agent_expected_format_correct"] += int(expected_format_correct)
            # Routing is about whether the model attempted an external action.
            # Parseability is a separate realization metric: conflating the two
            # made legacy results on tool-heavy suites look like better routing.
            strict_decision_correct = (
                attempted_tool if expected_tool is not None else not attempted_tool
            )
            parseable_decision_correct = (
                bool(predicted_calls) if expected_tool is not None else not attempted_tool
            )
            totals["agent_tool_decision_correct"] += int(strict_decision_correct)
            totals["agent_parseable_tool_decision_correct"] += int(parseable_decision_correct)
            totals["agent_regex_tool_decision_correct"] += int(
                (regex_predicted_tool is None) == (expected_tool is None)
            )
            predicted_arguments: dict[str, Any] = {}
            expected_arguments: dict[str, Any] = {}
            arguments_valid = False
            argument_keys_correct = False
            argument_values_exact = False
            required_values_exact = False
            required_values_normalized = False
            required_values_elastic = False
            argument_fields_expected = 0
            argument_fields_exact = 0
            required_fields_expected = 0
            required_fields_exact = 0
            arguments_typed = False
            direct_criteria_pass = False
            human_task_success = False
            strict_human_task_success = False
            elastic_human_task_success = False
            if expected_tool is not None:
                totals["agent_tool_required_samples"] += 1
                totals["agent_tool_true_positive"] += int(attempted_tool)
                totals["agent_parseable_tool_true_positive"] += int(bool(predicted_calls))
                totals["agent_tool_name_expected"] += 1
                totals["agent_tool_name_correct"] += int(predicted_tool == expected_tool)
                totals["agent_selected_tool_name_correct"] += int(
                    regex_predicted_tool == expected_tool
                )
                per_tool_expected[expected_tool] += 1
                per_tool_correct[expected_tool] += int(predicted_tool == expected_tool)
                if expected_format_correct:
                    totals["agent_tool_name_format_eligible"] += 1
                    totals["agent_tool_name_correct_given_expected_format"] += int(
                        predicted_tool == expected_tool
                    )
                if predicted_calls:
                    try:
                        predicted_arguments = json.loads(
                            predicted_calls[0]["function"]["arguments"]
                        )
                    except (KeyError, TypeError, json.JSONDecodeError):
                        predicted_arguments = {}
                if expected_calls:
                    try:
                        expected_arguments = json.loads(expected_calls[0]["function"]["arguments"])
                    except (KeyError, TypeError, json.JSONDecodeError):
                        expected_arguments = {}
                required = REQUIRED_ARGUMENTS.get(expected_tool, set())
                arguments_valid = predicted_tool == expected_tool and required.issubset(
                    set(predicted_arguments)
                )
                totals["agent_tool_arguments_valid"] += int(arguments_valid)
                expected_types = ARGUMENT_TYPES.get(expected_tool, {})
                arguments_typed = (
                    predicted_tool == expected_tool
                    and required.issubset(set(predicted_arguments))
                    and all(
                        isinstance(predicted_arguments.get(key), expected_type)
                        for key, expected_type in expected_types.items()
                        if key in predicted_arguments
                    )
                )
                totals["agent_tool_arguments_typed"] += int(arguments_typed)
                argument_keys_correct = predicted_tool == expected_tool and set(
                    predicted_arguments
                ) == set(expected_arguments)
                totals["agent_tool_argument_keys_correct"] += int(argument_keys_correct)
                argument_values_exact = (
                    predicted_tool == expected_tool and predicted_arguments == expected_arguments
                )
                totals["agent_tool_argument_values_exact"] += int(argument_values_exact)
                expected_required = {
                    key: expected_arguments[key] for key in required if key in expected_arguments
                }
                predicted_required = {
                    key: predicted_arguments[key]
                    for key in expected_required
                    if key in predicted_arguments
                }
                required_values_exact = (
                    predicted_tool == expected_tool
                    and predicted_required == expected_required
                    and len(predicted_required) == len(expected_required)
                )
                required_values_normalized = (
                    predicted_tool == expected_tool
                    and len(predicted_required) == len(expected_required)
                    and all(
                        _diagnostic_normalize_value(key, predicted_required[key])
                        == _diagnostic_normalize_value(key, expected_required[key])
                        for key in expected_required
                    )
                )
                required_values_elastic = (
                    predicted_tool == expected_tool
                    and len(predicted_required) == len(expected_required)
                    and all(
                        _elastic_values_equal(key, predicted_required[key], expected_required[key])
                        for key in expected_required
                    )
                )
                totals["agent_required_argument_calls_exact"] += int(required_values_exact)
                totals["agent_required_argument_calls_normalized"] += int(
                    required_values_normalized
                )
                totals["agent_required_argument_calls_elastic"] += int(required_values_elastic)
                argument_fields_expected = len(expected_arguments)
                argument_fields_exact = sum(
                    predicted_tool == expected_tool
                    and key in predicted_arguments
                    and predicted_arguments[key] == value
                    for key, value in expected_arguments.items()
                )
                required_fields_expected = len(expected_required)
                required_fields_exact = sum(
                    predicted_tool == expected_tool
                    and key in predicted_arguments
                    and predicted_arguments[key] == value
                    for key, value in expected_required.items()
                )
                totals["agent_argument_fields_expected"] += argument_fields_expected
                totals["agent_argument_fields_exact"] += argument_fields_exact
                totals["agent_required_argument_fields_expected"] += required_fields_expected
                totals["agent_required_argument_fields_exact"] += required_fields_exact
                per_tool_whole_exact[expected_tool] += int(argument_values_exact)
                per_tool_required_exact[expected_tool] += int(required_values_exact)
                per_tool_required_elastic[expected_tool] += int(required_values_elastic)
                per_tool_fields_expected[expected_tool] += argument_fields_expected
                per_tool_fields_exact[expected_tool] += argument_fields_exact
                per_tool_required_fields_expected[expected_tool] += required_fields_expected
                per_tool_required_fields_exact[expected_tool] += required_fields_exact
                totals["agent_exact_call_count"] += int(len(predicted_calls) == len(expected_calls))
                human_task_success = required_values_exact
                strict_human_task_success = argument_values_exact
                elastic_human_task_success = required_values_elastic
            else:
                totals["agent_no_tool_samples"] += 1
                totals["agent_no_tool_true_negative"] += int(not attempted_tool)
                totals["agent_nonempty_no_tool_answer"] += int(bool(predicted_content.strip()))
                totals["agent_no_tool_answer_exact"] += int(
                    predicted_content.strip() == expected.strip()
                )
                required_terms = [
                    str(value).strip().lower()
                    for value in evaluation_metadata.get("required_terms", [])
                    if str(value).strip()
                ]
                if required_terms:
                    totals["agent_direct_criteria_samples"] += 1
                    normalized_content = " ".join(predicted_content.lower().split())
                    direct_criteria_pass = bool(
                        not attempted_tool
                        and all(term in normalized_content for term in required_terms)
                    )
                    totals["agent_direct_required_terms_pass"] += int(direct_criteria_pass)
                else:
                    direct_criteria_pass = bool(not attempted_tool and predicted_content.strip())
                human_task_success = direct_criteria_pass
                strict_human_task_success = direct_criteria_pass
                elastic_human_task_success = direct_criteria_pass
            totals["agent_human_task_success"] += int(human_task_success)
            totals["agent_strict_human_task_success"] += int(strict_human_task_success)
            totals["agent_elastic_human_task_success"] += int(elastic_human_task_success)
            if include_predictions:
                predictions.append(
                    {
                        "index": index,
                        "evaluation_metadata": evaluation_metadata,
                        "expected": expected,
                        "prediction": prediction,
                        "expected_tool": expected_tool,
                        "predicted_tool": predicted_tool,
                        "regex_predicted_tool": regex_predicted_tool,
                        "expected_format_correct": expected_format_correct,
                        "tool_decision_correct": strict_decision_correct,
                        "parseable_tool_decision_correct": parseable_decision_correct,
                        "attempted_tool": attempted_tool,
                        "tool_name_correct": predicted_tool == expected_tool,
                        "selected_tool_name_correct": regex_predicted_tool == expected_tool,
                        "expected_arguments": expected_arguments,
                        "predicted_arguments": predicted_arguments,
                        "arguments_valid": arguments_valid,
                        "arguments_typed": arguments_typed,
                        "argument_keys_correct": argument_keys_correct,
                        "argument_values_exact": argument_values_exact,
                        "required_argument_values_exact": required_values_exact,
                        "required_argument_values_normalized": required_values_normalized,
                        "required_argument_values_elastic": required_values_elastic,
                        "argument_fields_expected": argument_fields_expected,
                        "argument_fields_exact": argument_fields_exact,
                        "required_argument_fields_expected": required_fields_expected,
                        "required_argument_fields_exact": required_fields_exact,
                        "direct_required_terms_pass": direct_criteria_pass,
                        "human_task_success": human_task_success,
                        "strict_human_task_success": strict_human_task_success,
                        "elastic_human_task_success": elastic_human_task_success,
                    }
                )
    finally:
        model.config.use_cache = old_cache
        model.train(was_training)
    count = totals["agent_eval_samples"]
    expected_count = totals["agent_tool_name_expected"]
    totals["agent_format_valid_rate"] = totals["agent_format_valid"] / max(1, count)
    totals["agent_expected_format_accuracy"] = totals["agent_expected_format_correct"] / max(
        1, count
    )
    totals["agent_tool_decision_accuracy"] = totals["agent_tool_decision_correct"] / max(1, count)
    totals["agent_regex_tool_decision_accuracy"] = totals[
        "agent_regex_tool_decision_correct"
    ] / max(1, count)
    tool_recall = totals["agent_tool_true_positive"] / max(1, totals["agent_tool_required_samples"])
    parseable_tool_recall = totals["agent_parseable_tool_true_positive"] / max(
        1, totals["agent_tool_required_samples"]
    )
    no_tool_specificity = totals["agent_no_tool_true_negative"] / max(
        1, totals["agent_no_tool_samples"]
    )
    totals["agent_tool_recall"] = tool_recall
    totals["agent_parseable_tool_recall"] = parseable_tool_recall
    totals["agent_no_tool_specificity"] = no_tool_specificity
    totals["agent_balanced_tool_decision_accuracy"] = 0.5 * (tool_recall + no_tool_specificity)
    totals["agent_parseable_balanced_tool_decision_accuracy"] = 0.5 * (
        parseable_tool_recall + no_tool_specificity
    )
    tp = totals["agent_tool_true_positive"]
    fn = totals["agent_tool_required_samples"] - tp
    tn = totals["agent_no_tool_true_negative"]
    fp = totals["agent_no_tool_samples"] - tn
    denominator = math.sqrt(max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    totals["agent_tool_decision_mcc"] = (tp * tn - fp * fn) / denominator
    totals["agent_tool_name_accuracy"] = totals["agent_tool_name_correct"] / max(1, expected_count)
    totals["agent_selected_tool_name_accuracy"] = totals["agent_selected_tool_name_correct"] / max(
        1, expected_count
    )
    totals["agent_tool_name_accuracy_given_expected_format"] = totals[
        "agent_tool_name_correct_given_expected_format"
    ] / max(1, totals["agent_tool_name_format_eligible"])
    totals["agent_tool_arguments_valid_rate"] = totals["agent_tool_arguments_valid"] / max(
        1, expected_count
    )
    totals["agent_tool_argument_keys_accuracy"] = totals["agent_tool_argument_keys_correct"] / max(
        1, expected_count
    )
    totals["agent_tool_argument_exact_accuracy"] = totals["agent_tool_argument_values_exact"] / max(
        1, expected_count
    )
    totals["agent_required_argument_call_exact_accuracy"] = totals[
        "agent_required_argument_calls_exact"
    ] / max(1, expected_count)
    totals["agent_required_argument_call_normalized_accuracy"] = totals[
        "agent_required_argument_calls_normalized"
    ] / max(1, expected_count)
    totals["agent_required_argument_call_elastic_accuracy"] = totals[
        "agent_required_argument_calls_elastic"
    ] / max(1, expected_count)
    totals["agent_argument_field_micro_accuracy"] = totals["agent_argument_fields_exact"] / max(
        1, totals["agent_argument_fields_expected"]
    )
    totals["agent_required_argument_field_micro_accuracy"] = totals[
        "agent_required_argument_fields_exact"
    ] / max(1, totals["agent_required_argument_fields_expected"])
    totals["agent_tool_arguments_typed_rate"] = totals["agent_tool_arguments_typed"] / max(
        1, expected_count
    )
    totals["agent_no_tool_nonempty_rate"] = totals["agent_nonempty_no_tool_answer"] / max(
        1, totals["agent_no_tool_samples"]
    )
    totals["agent_no_tool_answer_exact_rate"] = totals["agent_no_tool_answer_exact"] / max(
        1, totals["agent_no_tool_samples"]
    )
    totals["agent_direct_required_terms_accuracy"] = totals[
        "agent_direct_required_terms_pass"
    ] / max(1, totals["agent_direct_criteria_samples"])
    totals["agent_human_task_success_rate"] = totals["agent_human_task_success"] / max(1, count)
    totals["agent_strict_human_task_success_rate"] = totals[
        "agent_strict_human_task_success"
    ] / max(1, count)
    totals["agent_elastic_human_task_success_rate"] = totals[
        "agent_elastic_human_task_success"
    ] / max(1, count)
    totals["agent_tool_schema_applied"] = schema_applied
    totals["agent_tool_menu_names"] = [item["function"]["name"] for item in schema_menu]
    totals["agent_per_tool"] = {
        name: {
            "expected": per_tool_expected[name],
            "correct": per_tool_correct[name],
            "accuracy": per_tool_correct[name] / max(1, per_tool_expected[name]),
            "whole_call_exact": per_tool_whole_exact[name],
            "whole_call_exact_accuracy": per_tool_whole_exact[name]
            / max(1, per_tool_expected[name]),
            "required_call_exact": per_tool_required_exact[name],
            "required_call_exact_accuracy": per_tool_required_exact[name]
            / max(1, per_tool_expected[name]),
            "required_call_elastic": per_tool_required_elastic[name],
            "required_call_elastic_accuracy": per_tool_required_elastic[name]
            / max(1, per_tool_expected[name]),
            "field_micro_accuracy": per_tool_fields_exact[name]
            / max(1, per_tool_fields_expected[name]),
            "required_field_micro_accuracy": per_tool_required_fields_exact[name]
            / max(1, per_tool_required_fields_expected[name]),
        }
        for name in sorted(per_tool_expected)
    }
    if include_predictions:
        totals["agent_predictions"] = predictions
        totals["agent_slices"] = evaluation_slices_from_predictions(predictions)
    return totals
