"""Thin adapter around pinned official BFCL packages.

This file intentionally does not reimplement BFCL metrics.  It only registers a local
OpenAI-compatible model handler and delegates generation/evaluation to the official code.
"""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

LFM2_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of <|tool_call_start|>[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]<|tool_call_end|>
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.
"""
LFM2_AGENTIC_FORMAT_INSTRUCTION = (
    "When you decide to invoke any of the available function(s), you MUST put the "
    "call(s) in the format of <|tool_call_start|>[func_name1(params_name1=params_value1, "
    "params_name2=params_value2...), func_name2(params)]<|tool_call_end|>."
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _prepend_liquid_instruction(
    question: list[list[dict[str, Any]]], *, agentic: bool = False
) -> None:
    if not question:
        return
    first = question[0]
    instruction = LFM2_AGENTIC_FORMAT_INSTRUCTION if agentic else LFM2_SYSTEM_PROMPT
    if first and first[0].get("role") == "system":
        existing = str(first[0].get("content") or "")
        first[0]["content"] = (
            existing + "\n\n" + instruction
            if agentic
            else instruction + "\n\n" + existing
        )
    else:
        first.insert(0, {"role": "system", "content": instruction})


def _install_compact_v4_model_registry() -> tuple[dict[str, Any], type]:
    """Replace BFCL's eager all-vendor registry with an API-only compatible registry.

    Official BFCL imports every optional vendor and CUDA handler from model_config.py. The
    evaluator only needs this dataclass and mapping. Providing the same interface avoids several
    gigabytes of irrelevant dependencies while leaving official datasets/checkers untouched.
    """

    @dataclass
    class CompactModelConfig:
        model_name: str
        display_name: str
        url: str
        org: str
        license: str
        model_handler: Any
        input_price: float | None = None
        output_price: float | None = None
        is_fc_model: bool = True
        underscore_to_dot: bool = False

    mapping: dict[str, Any] = {}
    module = ModuleType("bfcl_eval.constants.model_config")
    module.ModelConfig = CompactModelConfig
    module.MODEL_CONFIG_MAPPING = mapping
    sys.modules[module.__name__] = module
    return mapping, CompactModelConfig


def _install_compact_v3_handler_registry() -> dict[str, Any]:
    """Replace BFCL v3's eager all-vendor handler registry.

    The pinned v3 ``handler_map`` imports every proprietary SDK at module import time,
    including Anthropic, Cohere and Gemini.  A managed local run needs only the OpenAI-
    compatible handler registered below, so exposing the same ``handler_map`` interface
    keeps the official generator/evaluator intact without installing unrelated vendor
    clients (or their transitive disk footprint).
    """

    mapping: dict[str, Any] = {}
    module = ModuleType("bfcl.model_handler.handler_map")
    module.handler_map = mapping
    sys.modules[module.__name__] = module
    return mapping


def _run_v4(config: dict[str, Any], source: Path, endpoint: str) -> dict[str, Any]:
    output = Path(config["output_dir"])
    os.environ["BFCL_PROJECT_ROOT"] = str(output)
    os.environ["OPENAI_BASE_URL"] = endpoint.rstrip("/")
    os.environ["OPENAI_API_KEY"] = "EMPTY"

    sys.path.insert(0, str(source))
    model_mapping, model_config_type = _install_compact_v4_model_registry()
    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from bfcl_eval.utils import extract_test_category_from_id, is_agentic, load_dataset_entry

    class SafeOpenAICompletionsHandler(OpenAICompletionsHandler):
        def decode_execute(self, result, has_tool_call_tag):
            if isinstance(result, str) or not result:
                return []
            return super().decode_execute(result, has_tool_call_tag)

    class LiquidAPIHandler(SafeOpenAICompletionsHandler):
        def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
            category = extract_test_category_from_id(test_entry["id"])
            _prepend_liquid_instruction(test_entry["question"], agentic=is_agentic(category))
            return super()._pre_query_processing_FC(inference_data, test_entry)

    registry_name = "exotic-local-liquid-fc" if config["handler"] == "liquid-lfm2" else "exotic-local-generic-fc"
    handler = LiquidAPIHandler if config["handler"] == "liquid-lfm2" else SafeOpenAICompletionsHandler
    model_mapping[registry_name] = model_config_type(
        model_name=str(config.get("api_model") or "local-model"),
        display_name=str(config.get("target_name") or registry_name),
        url="local://exotic-agent-trainer",
        org="Local",
        license="local",
        model_handler=handler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )

    categories = list(config["categories"])
    run_ids = bool(config.get("partial"))
    selected_count = None
    if run_ids:
        configured_id_map = config.get("test_case_ids")
        if configured_id_map is not None:
            if not isinstance(configured_id_map, dict) or not configured_id_map:
                raise ValueError("test_case_ids must be a non-empty category-to-ID mapping")
            id_map = {}
            for category, identifiers in configured_id_map.items():
                if not isinstance(category, str) or not isinstance(identifiers, list):
                    raise ValueError("test_case_ids must map category strings to ID lists")
                available = {str(item["id"]) for item in load_dataset_entry(category)}
                requested = [str(identifier) for identifier in identifiers]
                missing = sorted(set(requested) - available)
                if missing:
                    raise ValueError(
                        f"Unknown BFCL IDs for {category}: {missing[:5]}"
                    )
                id_map[category] = requested
            categories = sorted(id_map)
        else:
            concrete = categories
            budget = max(2, int(config.get("smoke_samples") or 20))
            per_category = max(1, budget // len(concrete))
            id_map = {}
            for category in concrete:
                entries = load_dataset_entry(category)
                id_map[category] = [str(item["id"]) for item in entries[:per_category]]
        selected_count = sum(len(values) for values in id_map.values())
        _atomic_json(output / "test_case_ids_to_generate.json", id_map)
        _atomic_json(
            output / "official-progress.json",
            {
                "stage": "bfcl-generation",
                "total_cases": selected_count,
                "selected_categories": len(id_map),
            },
        )

    generation_args = SimpleNamespace(
        model=[registry_name],
        test_category=categories,
        temperature=0.001,
        include_input_log=True,
        exclude_state_log=False,
        num_gpus=0,
        num_threads=1,
        gpu_memory_utilization=0.0,
        backend="vllm",
        skip_server_setup=True,
        local_model_path=None,
        result_dir="result",
        allow_overwrite=True,
        run_ids=run_ids,
        enable_lora=False,
        max_lora_rank=None,
        lora_modules=None,
    )
    generation_main(generation_args)
    evaluation_main(
        [registry_name],
        categories,
        "result",
        "score",
        partial_eval=run_ids,
    )
    return {
        "version": "v4",
        "registry_name": registry_name,
        "categories": categories,
        "partial": run_ids,
        "selected_samples": selected_count,
        "complete": not run_ids,
        "result_dir": str(output / "result"),
        "score_dir": str(output / "score"),
        "handler_provenance": config.get("handler_provenance"),
    }


def _install_v3_executable_retries(max_attempts: int = 6) -> list[str]:
    """Retry BFCL v3's live executable helpers without changing successful outputs.

    The released v3 scorer obtains some ground-truth values from third-party APIs.  A
    transient 429/error payload currently aborts the whole evaluation (for example, the
    Urban Dictionary helper indexes ``response.json()["list"]`` without validating the
    response).  Wrap only helpers whose source directly performs an HTTP request.  A
    successful first call is byte-for-byte the official behaviour; failures are retried
    and are still raised rather than replaced with synthetic ground truth.
    """

    from bfcl.eval_checker.executable_eval.data import executable_python_function as module

    wrapped: list[str] = []
    for name, function in list(vars(module).items()):
        if name.startswith("_") or not inspect.isfunction(function):
            continue
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            continue
        if "requests.get(" not in source and "requests.post(" not in source:
            continue

        @functools.wraps(function)
        def resilient(*args: Any, __function: Any = function, **kwargs: Any) -> Any:
            error: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return __function(*args, **kwargs)
                except Exception as current_error:
                    error = current_error
                    if attempt + 1 < max_attempts:
                        time.sleep(min(2.0, 0.5 * (attempt + 1)))
            return f"API Execution Error: {error}"

        setattr(module, name, resilient)
        wrapped.append(name)
    return sorted(wrapped)


def _run_v3(
    config: dict[str, Any],
    source: Path,
    endpoint: str,
    *,
    score_only: bool = False,
    score_categories: list[str] | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    output = Path(config["output_dir"]).resolve()
    from exotic_trainer.credentials import load_bfcl_environment

    load_bfcl_environment()
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    sys.path.insert(0, str(source))
    os.chdir(output)
    data_source = source / "data"
    configured_data = config.get("data_dir")
    if configured_data:
        data_source = Path(str(configured_data)).resolve()
        manifest_path = data_source / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"BFCL v3 subset manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("bfcl_commit") != config.get("bfcl_commit"):
            raise ValueError("BFCL v3 subset commit does not match the pinned evaluator")
        if not manifest.get("partial"):
            raise ValueError("Configured BFCL v3 data_dir is not marked partial")
    data_link = output / "data"
    if not data_link.exists():
        data_link.symlink_to(data_source, target_is_directory=True)
    # BFCL v3's executable evaluator reads this directory through a hard-coded
    # cwd-relative path at import time.  Link the pinned official data rather
    # than copying it into every run directory.
    executable_eval_link = output / "executable_eval"
    if not executable_eval_link.exists():
        executable_eval_link.symlink_to(
            source / "bfcl" / "eval_checker" / "executable_eval",
            target_is_directory=True,
        )
    (output / "result").mkdir(exist_ok=True)
    (output / "score").mkdir(exist_ok=True)

    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    handler_map = _install_compact_v3_handler_registry()
    import openfunctions_evaluation as generation
    from bfcl.constant import TEST_COLLECTION_MAPPING
    from bfcl.eval_checker import eval_runner as evaluation
    from bfcl.eval_checker import eval_runner_helper
    from bfcl.model_handler.proprietary_model.openai import OpenAIHandler
    from openai import OpenAI

    class GenericV3Handler(OpenAIHandler):
        def __init__(self, model_name: str, temperature: float) -> None:
            super().__init__(model_name, temperature)
            limits = config.get("runtime_limits") or {}
            self.client = OpenAI(
                base_url=endpoint.rstrip("/"),
                api_key="EMPTY",
                timeout=float(limits.get("request_timeout_seconds") or 210.0),
                max_retries=int(limits.get("openai_max_retries") or 0),
            )

        def inference(self, test_entry: dict, include_debugging_log: bool) -> tuple[Any, dict]:
            """Dispatch the pinned v3 dataset without double-wrapping single turns.

            The released BFCL v3 JSON already stores ``question`` as a list of turns,
            including for single-turn categories.  Its base dispatcher wraps that list
            once more, producing a shape that none of the official chat handlers accept.
            The local target is always an FC model, so dispatch directly while preserving
            the official single- and multi-turn implementations below.
            """

            if "multi_turn" in str(test_entry["id"]):
                return self.inference_multi_turn_FC(test_entry, include_debugging_log)
            return self.inference_single_turn_FC(test_entry, include_debugging_log)

        def inference_single_turn_FC(
            self, test_entry: dict, include_debugging_log: bool
        ) -> tuple[Any, dict[str, Any]]:
            """Bridge a signature regression in the pinned official BFCL v3 runner.

            At the pinned v3 commit, ``BaseHandler.inference_single_turn_FC`` calls
            ``_pre_query_processing_FC`` with only ``test_entry`` even though the base
            contract and every API handler require ``(inference_data, test_entry)``.
            That branch also omits the tool-compilation step.  Reproduce the official
            single-turn flow with those two omissions restored; parsing, decoding,
            datasets and scoring remain the pinned official implementation.
            """

            inference_data: dict[str, Any] = {}
            inference_data = self._pre_query_processing_FC(inference_data, test_entry)
            inference_data = self._compile_tools(inference_data, test_entry)
            inference_data = self.add_first_turn_message_FC(
                inference_data, test_entry["question"][0]
            )

            started = time.time()
            api_response = self._query_FC(inference_data)
            query_latency = time.time() - started
            model_response_data = self._parse_query_response_FC(api_response)

            metadata: dict[str, Any] = {}
            if include_debugging_log:
                metadata["debugging_log"] = [
                    {
                        "role": "handler_log:inference_input",
                        "content": inference_data["inference_input_log"],
                    }
                ]
            metadata["input_token_count"] = model_response_data["input_token"]
            metadata["output_token_count"] = model_response_data["output_token"]
            metadata["latency"] = query_latency
            return model_response_data["model_responses"], metadata

    class LiquidV3Handler(GenericV3Handler):
        def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
            _prepend_liquid_instruction(test_entry["question"])
            return super()._pre_query_processing_FC(inference_data, test_entry)

    registry_name = "exotic-local-liquid-FC" if config["handler"] == "liquid-lfm2" else "exotic-local-generic-FC"
    handler_map[registry_name] = LiquidV3Handler if config["handler"] == "liquid-lfm2" else GenericV3Handler
    # The official CSV aggregator assumes every evaluated model is present in the
    # release metadata table.  Register the local target so score-only recovery can
    # finish aggregation instead of raising KeyError after all categories are scored.
    eval_runner_helper.MODEL_METADATA_MAPPING.setdefault(
        registry_name,
        [
            str(config.get("target_name") or registry_name),
            "local://exotic-agent-trainer",
            "Local",
            "Local evaluation",
        ],
    )
    _categories, files = generation.parse_test_category_argument(list(config["categories"]))
    tests = generation.collect_test_cases(files, registry_name)
    if score_only:
        generated = sum(
            1
            for path in (output / "result").rglob("*.json")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if generated == 0:
            raise RuntimeError(
                f"score-only requires predictions, but found 0 in {output / 'result'}"
            )
    else:
        _atomic_json(
            output / "official-progress.json",
            {
                "stage": "compatibility-probe",
                "total_cases": len(tests),
                "categories": _categories,
            },
        )
        # Exercise one real official case before scheduling the configured run.
        # BFCL v3 records most handler exceptions as ordinary wrong answers, which can
        # otherwise make an integration failure look like a very fast valid benchmark.
        probe_handler = handler_map[registry_name](registry_name, 0.001)
        probe_result, probe_metadata = probe_handler.inference(deepcopy(tests[0]), True)
        _atomic_json(
            output / "compatibility-probe.json",
            {
                "id": tests[0]["id"],
                "result": probe_result,
                "metadata": probe_metadata,
            },
        )
        _atomic_json(
            output / "official-progress.json",
            {
                "stage": "generation",
                "total_cases": len(tests),
                "categories": _categories,
            },
        )
        args = SimpleNamespace(
            temperature=0.001,
            include_debugging_log=True,
            num_threads=1,
            num_gpus=0,
            gpu_memory_utilization=0.0,
        )
        generation.generate_results(args, registry_name, tests)

    _atomic_json(
        output / "official-progress.json",
        {
            "stage": "evaluation",
            "total_cases": len(tests),
            "generated_cases": len(tests),
            "categories": _categories,
        },
    )

    evaluation.INPUT_PATH = str(output / "result") + "/"
    # The released scorer temporarily writes live execution results into its dataset.
    # Work on a compact run-local copy so a crash cannot contaminate the pinned source.
    # Keep protocol metadata outside the directory scanned by the pinned evaluator:
    # its file discovery assumes every top-level JSON file is named BFCL_v3_<category>.json.
    scoring_data = output / "scoring-data-v3-evaluator"
    if not scoring_data.exists():
        shutil.copytree(
            data_source,
            scoring_data,
            ignore=shutil.ignore_patterns("manifest.json"),
        )
    evaluation.PROMPT_PATH = str(scoring_data) + "/"
    evaluation.POSSIBLE_ANSWER_PATH = str(scoring_data / "possible_answer") + "/"
    evaluation.OUTPUT_PATH = str(output / "score") + "/"
    expanded: list[str] = []
    for category in score_categories or config["categories"]:
        expanded.extend(TEST_COLLECTION_MAPPING.get(category, [category]))
    if any(category.startswith("exec_") or category == "rest" for category in expanded):
        load_bfcl_environment(require_live=True)
    retry_helpers = _install_v3_executable_retries()
    evaluation.runner([registry_name], sorted(set(expanded)), api_sanity_check=False)
    _atomic_json(
        output / "official-progress.json",
        {
            "stage": "complete",
            "total_cases": len(tests),
            "generated_cases": len(tests),
            "categories": _categories,
        },
    )
    partial = bool(config.get("partial"))
    return {
        "version": "v3",
        "registry_name": registry_name,
        "categories": sorted(set(expanded)),
        "partial": partial,
        "selected_samples": len(tests) if partial else None,
        "complete": not partial,
        "result_dir": str(output / "result"),
        "score_dir": str(output / "score"),
        "handler_provenance": config.get("handler_provenance"),
        "score_only": score_only,
        "executable_retry_helpers": retry_helpers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9/v1")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--score-category", action="append", default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = (
        _run_v4(deepcopy(config), args.source, args.endpoint)
        if config["version"] == "v4"
        else _run_v3(
            deepcopy(config),
            args.source,
            args.endpoint,
            score_only=args.score_only,
            score_categories=args.score_category,
        )
    )
    _atomic_json(Path(config["output_dir"]) / "official-result.json", result)


if __name__ == "__main__":
    main()
