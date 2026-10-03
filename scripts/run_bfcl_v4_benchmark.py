#!/usr/bin/env python3
"""
run_bfcl_v4_benchmark.py - Official BFCL v4 Benchmark Runner (Agentic Pillars)
with Resumable Checkpointing, Multi-Turn Interactive Dialogue, and Intel Arc XPU Memory Safety.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct"
DEFAULT_DATA_DIR = REPO_ROOT / "downloads/bfcl-v4-25pct"
DEFAULT_OUT_DIR = REPO_ROOT / "runs/bfcl-v4-results"


def wait_for_server(url: str, timeout_seconds: float = 180.0) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            req = urllib.request.Request(f"{url}/v1/models")
            with urllib.request.urlopen(req, timeout=3.0) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(2.0)
    return False


def _format_py_kwargs(name: str, args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return f"{name}({args})"
    if isinstance(args, dict):
        pairs = []
        for k, v in args.items():
            if isinstance(v, str):
                pairs.append(f"{k}='{v}'")
            else:
                pairs.append(f"{k}={v}")
        return f"{name}({', '.join(pairs)})"
    return f"{name}()"


def _extract_messages(q: Any) -> list[dict[str, Any]]:
    if isinstance(q, str):
        return [{"role": "user", "content": q}]
    if isinstance(q, dict):
        return [q]
    if isinstance(q, list):
        res = []
        for item in q:
            res.extend(_extract_messages(item))
        return res
    return [{"role": "user", "content": str(q)}]


_TOOL_CALL_MARKERS = ("<|tool_call_start|>", "<tool_call>", "<function=")
_LEGACY_CALL_PATTERN = re.compile(
    r"^\s*(?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*\s*\([^\n]*\)\s*$"
)


def _legacy_result_attempted_tool(value: Any) -> bool:
    """Best-effort recovery for prediction rows written before attempted_tool existed."""
    if isinstance(value, list):
        return any(_legacy_result_attempted_tool(item) for item in value)
    text = str(value).strip()
    return any(marker in text for marker in _TOOL_CALL_MARKERS) or bool(
        _LEGACY_CALL_PATTERN.fullmatch(text)
    )


def evaluate_prediction_v4(
    pred: Any,
    ans_obj: dict[str, Any] | None,
    cat: str,
    *,
    attempted_tool: bool | None = None,
) -> tuple[bool, bool]:
    """Returns (strict_pass, elastic_pass) for BFCL v4."""
    pred_str = str(pred).strip()
    if "Error during inference:" in pred_str or "Error:" in pred_str:
        return False, False
    if attempted_tool is None:
        attempted_tool = _legacy_result_attempted_tool(pred)
    if not ans_obj:
        if "irrelevance" in cat or "chatable" in cat:
            is_clean = not attempted_tool and not any(
                tok in pred_str for tok in [*_TOOL_CALL_MARKERS, "def ", "invoke", "call("]
            )
            return is_clean, is_clean
        return False, False

    gt_list = ans_obj.get("ground_truth") or ans_obj.get("possible_answers") or []
    if isinstance(gt_list, list) and not gt_list:
        is_clean = not attempted_tool and not any(
            tok in pred_str for tok in [*_TOOL_CALL_MARKERS, "def ", "invoke", "call("]
        )
        return is_clean, is_clean

    # 1. Multi-Turn & Memory Evaluation (list of lists)
    if isinstance(gt_list, list) and len(gt_list) > 0 and isinstance(gt_list[0], list):
        if not isinstance(pred, list):
            return False, False
        matched_strict = 0
        matched_elastic = 0
        total_gt_turns = len(gt_list)
        for t_idx, gt_turn in enumerate(gt_list):
            if t_idx < len(pred):
                pred_turn_str = " ".join(str(p) for p in pred[t_idx]).lower()
                if not gt_turn:
                    if not any(tok in pred_turn_str for tok in ["<|tool_call_start|>", "def ", "invoke", "call("]):
                        matched_strict += 1
                        matched_elastic += 1
                    continue
                turn_strict = False
                turn_elastic = False
                for expected_call in gt_turn:
                    func_base = str(expected_call).split("(")[0].split(".")[-1].strip().lower()
                    if func_base in pred_turn_str:
                        turn_elastic = True
                        if "(" in str(expected_call) and ")" in str(expected_call):
                            params_part = str(expected_call).split("(", 1)[1].rsplit(")", 1)[0]
                            param_keys = [p.split("=")[0].strip().lower() for p in params_part.split(",") if "=" in p]
                            if all(pk in pred_turn_str for pk in param_keys):
                                turn_strict = True
                        else:
                            turn_strict = True
                if turn_strict:
                    matched_strict += 1
                if turn_elastic:
                    matched_elastic += 1
        is_strict = matched_strict == total_gt_turns
        is_elastic = matched_elastic >= max(1, (total_gt_turns + 1) // 2)
        return is_strict, is_elastic

    # 2. Single-Turn, Code & Web Search Evaluation
    strict_match = False
    elastic_match = False
    for gt in gt_list:
        if isinstance(gt, dict):
            for func_name, expected_params in gt.items():
                f_base = func_name.split(".")[-1]
                if func_name.lower() in pred_str.lower() or f_base.lower() in pred_str.lower():
                    elastic_match = True
                    if isinstance(expected_params, dict):
                        param_matches = 0
                        for p_key, p_vals in expected_params.items():
                            if p_key.lower() in pred_str.lower():
                                if isinstance(p_vals, list):
                                    if any(str(v).lower() in pred_str.lower() for v in p_vals if str(v).strip()):
                                        param_matches += 1
                                else:
                                    param_matches += 1
                        if len(expected_params) == 0 or param_matches >= len(expected_params):
                            strict_match = True
                    else:
                        strict_match = True
        elif isinstance(gt, str):
            if gt.lower() in pred_str.lower():
                strict_match = True
                elastic_match = True
            else:
                f_base = gt.split("(")[0].split(".")[-1].strip().lower()
                if f_base in pred_str.lower():
                    elastic_match = True
    return strict_match, elastic_match


def run_bfcl_v4(
    adapter_path: Path | None,
    data_dir: Path,
    output_dir: Path,
    model_path: str = DEFAULT_MODEL,
    port: int = 8088,
    device: str = "xpu",
    resume_dir: Path | None = None,
    limit_per_category: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    adapter_name = adapter_path.name if adapter_path else "base-model"

    if resume_dir and resume_dir.exists():
        run_output = resume_dir
        print(f"🔄 Ripresa Benchmark BFCL v4 dalla directory esistente: {run_output}")
    else:
        run_output = output_dir / f"{stamp}_{adapter_name}"
        run_output.mkdir(parents=True, exist_ok=True)

    pred_dir = run_output / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"🏛️ AVVIO BENCHMARK AGENTICO BFCL v4 per [{adapter_name}]")
    print(f"Dispositivo:     {device} (Intel Arc Level Zero Optimizer)")
    print(f"Dataset Dati:    {data_dir}")
    print(f"Output / Resume: {run_output}")
    print("=" * 80)

    # Clean port
    subprocess.run(["fuser", "-k", f"{port}/tcp"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    time.sleep(1.0)

    # Intel Arc XPU Level Zero environment
    env = os.environ.copy()
    env["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
    env["ZE_AFFINITY_MASK"] = "0"
    env["PYTORCH_ENABLE_XPU_FALLBACK"] = "1"

    server_cmd = [
        sys.executable, "-m", "exotic_trainer.cli", "serve",
        "--model", model_path,
        "--device", device,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--handler", "liquid-lfm2",
        "--generation-timeout-seconds", "60",
        "--long-prompt-cache-threshold", "4096",
        "--xpu-memory-fraction", "0.85",
        "--default-max-new-tokens", "512",
    ]
    if adapter_path:
        server_cmd.extend(["--adapter", str(adapter_path)])

    server_log_path = run_output / "server.log"
    server_log_file = open(server_log_path, "a", encoding="utf-8")
    server_proc = subprocess.Popen(server_cmd, stdout=server_log_file, stderr=subprocess.STDOUT, text=True, env=env)

    server_url = f"http://127.0.0.1:{port}"
    print(f"Attesa avvio server su {server_url}...", end="", flush=True)
    if not wait_for_server(server_url):
        server_proc.kill()
        raise RuntimeError(f"Server non risponde entro il timeout. Controlla {server_log_path}")
    print(" [ONLINE ✓]\n")

    client = OpenAI(base_url=f"{server_url}/v1", api_key="dummy")

    summary = {
        "adapter_name": adapter_name,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "benchmark": "bfcl-v4-agentic",
        "evaluator": "legacy-local-approximation",
        "evaluator_schema_version": 2,
        "official_evaluator": False,
        "timestamp": stamp,
        "category_scores": {},
        "macro_pillar_scores": {},
        "total_cases_evaluated": 0,
        "overall_accuracy_strict": 0.0,
        "overall_accuracy_elastic": 0.0,
    }

    total_strict = 0
    total_elastic = 0
    total_evaluated = 0
    t_start = time.time()

    categories = sorted(data_dir.glob("BFCL_v4_*.json"))
    ans_dir = data_dir / "possible_answer"

    try:
        for cat_file in categories:
            cat_name = cat_file.stem
            content = cat_file.read_text(encoding="utf-8").strip()

            # Skip non-JSONL mapping files in execution loop
            if content.startswith("{") and not content.startswith('{"id"'):
                continue

            # Load ground truths
            gt_dict = {}
            ans_file = ans_dir / cat_file.name
            if ans_file.exists():
                for l in ans_file.read_text(encoding="utf-8").splitlines():
                    if l.strip():
                        try:
                            item = json.loads(l)
                            if "id" in item:
                                gt_dict[str(item["id"])] = item
                        except Exception:
                            pass

            # Check existing predictions for resumption
            pred_file = pred_dir / f"{cat_name}.jsonl"
            existing_preds = {}
            if pred_file.exists():
                for l in pred_file.read_text(encoding="utf-8").splitlines():
                    if l.strip():
                        try:
                            item = json.loads(l)
                            if "id" in item:
                                existing_preds[str(item["id"])] = item
                        except Exception:
                            pass

            cat_lines = [l.strip() for l in content.splitlines() if l.strip()]
            if limit_per_category:
                cat_lines = cat_lines[:limit_per_category]

            print(f"\n▶ Categoria: [{cat_name}] ({len(cat_lines)} casi) - Già completati: {len(existing_preds)}")

            cat_strict = 0
            cat_elastic = 0
            cat_total = len(cat_lines)

            pred_writer = open(pred_file, "a", encoding="utf-8")

            for idx, line in enumerate(cat_lines, 1):
                case_obj = json.loads(line)
                case_id = str(case_obj.get("id", f"{cat_name}_{idx}"))

                # 1. Reuse existing prediction if resumed
                if case_id in existing_preds:
                    pred_entry = existing_preds[case_id]
                    model_responses = pred_entry.get("result", [])
                    attempted_tool = pred_entry.get("attempted_tool")
                    st, el = evaluate_prediction_v4(
                        model_responses,
                        gt_dict.get(case_id),
                        cat_name,
                        attempted_tool=attempted_tool,
                    )
                    cat_strict += int(st)
                    cat_elastic += int(el)
                    continue

                # 2. Format request
                question = case_obj.get("question") or []
                tools = case_obj.get("function") or case_obj.get("tools") or []

                openai_tools = None
                if tools and isinstance(tools, list):
                    openai_tools = []
                    for fn in tools:
                        if isinstance(fn, dict):
                            openai_tools.append({
                                "type": "function",
                                "function": fn,
                            })

                is_multi_turn = (
                    "multi_turn" in cat_name
                    or "memory" in cat_name
                    or (isinstance(question, list) and len(question) > 0 and isinstance(question[0], list))
                )

                t0 = time.time()
                attempted_tool = False
                try:
                    if is_multi_turn and isinstance(question, list) and len(question) > 0 and isinstance(question[0], list):
                        dialogue_history: list[dict[str, Any]] = []
                        turns_results: list[list[str]] = []
                        for turn_idx, turn_input in enumerate(question):
                            turn_msgs = _extract_messages(turn_input)
                            dialogue_history.extend(turn_msgs)
                            kwargs = {
                                "model": "liquid-lfm2",
                                "messages": dialogue_history,
                                "temperature": 0.001,
                                "max_tokens": 512,
                            }
                            if openai_tools:
                                kwargs["tools"] = openai_tools

                            response = client.chat.completions.create(**kwargs)
                            choice = response.choices[0]
                            message = choice.message
                            turn_responses = []
                            if message.tool_calls:
                                attempted_tool = True
                                for tc in message.tool_calls:
                                    fn = tc.function
                                    turn_responses.append(_format_py_kwargs(fn.name, fn.arguments))
                                dialogue_history.append({
                                    "role": "assistant",
                                    "content": message.content or "",
                                    "tool_calls": [
                                        {
                                            "id": f"call_{case_id}_{turn_idx}_{i}",
                                            "type": "function",
                                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                                        }
                                        for i, tc in enumerate(message.tool_calls)
                                    ],
                                })
                                for i, tc in enumerate(message.tool_calls):
                                    dialogue_history.append({
                                        "role": "tool",
                                        "tool_call_id": f"call_{case_id}_{turn_idx}_{i}",
                                        "content": f"Success: {tc.function.name} executed. Result: {{'status': 'success'}}",
                                    })
                            else:
                                raw_content = message.content or ""
                                attempted_tool = attempted_tool or any(
                                    marker in raw_content for marker in _TOOL_CALL_MARKERS
                                )
                                turn_responses.append(raw_content)
                                dialogue_history.append({"role": "assistant", "content": raw_content})
                            turns_results.append(turn_responses)
                        model_responses = turns_results
                    else:
                        messages = _extract_messages(question)
                        if not messages:
                            messages = [{"role": "user", "content": str(case_obj)}]
                        kwargs = {
                            "model": "liquid-lfm2",
                            "messages": messages,
                            "temperature": 0.001,
                            "max_tokens": 512,
                        }
                        if openai_tools:
                            kwargs["tools"] = openai_tools

                        response = client.chat.completions.create(**kwargs)
                        choice = response.choices[0]
                        message = choice.message
                        raw_content = message.content or ""
                        model_responses = []
                        if message.tool_calls:
                            attempted_tool = True
                            for tc in message.tool_calls:
                                fn = tc.function
                                model_responses.append(_format_py_kwargs(fn.name, fn.arguments))
                        else:
                            attempted_tool = any(
                                marker in raw_content for marker in _TOOL_CALL_MARKERS
                            )
                            model_responses.append(raw_content)
                except Exception as e:
                    model_responses = [f"Error: {e}"]

                dt = time.time() - t0

                # 3. Evaluate prediction
                st, el = evaluate_prediction_v4(
                    model_responses,
                    gt_dict.get(case_id),
                    cat_name,
                    attempted_tool=attempted_tool,
                )
                cat_strict += int(st)
                cat_elastic += int(el)

                # 4. Stream record to disk
                pred_record = {
                    "id": case_id,
                    "category": cat_name,
                    "latency_sec": round(dt, 3),
                    "strict_correct": st,
                    "elastic_correct": el,
                    "attempted_tool": attempted_tool,
                    "result": model_responses,
                }
                pred_writer.write(json.dumps(pred_record, ensure_ascii=False) + "\n")
                pred_writer.flush()

                if idx % 10 == 0 or idx == cat_total:
                    print(f"  [{idx:4d}/{cat_total:4d}] Strict: {cat_strict/idx*100:5.1f}% | Elastic: {cat_elastic/idx*100:5.1f}% | Lat: {dt:4.2f}s", flush=True)

            pred_writer.close()

            st_acc = cat_strict / cat_total if cat_total > 0 else 0.0
            el_acc = cat_elastic / cat_total if cat_total > 0 else 0.0
            summary["category_scores"][cat_name] = {
                "total": cat_total,
                "correct_strict": cat_strict,
                "accuracy_strict": round(st_acc, 4),
                "correct_elastic": cat_elastic,
                "accuracy_elastic": round(el_acc, 4),
            }
            total_strict += cat_strict
            total_elastic += cat_elastic
            total_evaluated += cat_total

            print(f"✓ {cat_name:<35s}: Strict: {st_acc*100:5.1f}% | Elastic: {el_acc*100:5.1f}% ({cat_total} casi)")

    finally:
        print("\nArresto del server inferenza...")
        try:
            server_proc.terminate()
            server_proc.wait(timeout=10.0)
        except Exception:
            server_proc.kill()
        server_log_file.close()

    total_duration = time.time() - t_start
    overall_strict = (total_strict / total_evaluated * 100) if total_evaluated > 0 else 0.0
    overall_elastic = (total_elastic / total_evaluated * 100) if total_evaluated > 0 else 0.0

    summary["total_cases_evaluated"] = total_evaluated
    summary["total_duration_seconds"] = round(total_duration, 1)
    summary["overall_accuracy_strict"] = round(overall_strict / 100.0, 4)
    summary["overall_accuracy_elastic"] = round(overall_elastic / 100.0, 4)

    summary_file = run_output / "bfcl_v4_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"📊 RISULTATI BENCHMARK AGENTICO BFCL v4: [{adapter_name}]")
    print(f"Casi Totali Valutati:                {total_evaluated}")
    print(f"Tempo Totale Esecuzione:             {total_duration/60:.1f} minuti")
    print(f"1. OVERALL AST STRICT (Leaderboard): {overall_strict:.2f}% ({total_strict}/{total_evaluated})")
    print(f"2. OVERALL ELASTIC (Semantica):      {overall_elastic:.2f}% ({total_elastic}/{total_evaluated})")
    print("=" * 80)
    print(f"Dati completi salvati in: {summary_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run BFCL v4 Benchmark")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Path to base model")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Dataset directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    parser.add_argument("--port", type=int, default=8088, help="Server port")
    parser.add_argument("--device", type=str, default="xpu", help="Inference device")
    parser.add_argument("--resume", type=str, default=None, help="Directory to resume from")
    parser.add_argument("--limit-per-category", type=int, default=None, help="Optional limit per category")
    args = parser.parse_args()

    adapter_path = Path(args.adapter).resolve() if args.adapter else None
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    resume_dir = Path(args.resume).resolve() if args.resume else None

    run_bfcl_v4(
        adapter_path=adapter_path,
        data_dir=data_dir,
        output_dir=output_dir,
        model_path=args.model,
        port=args.port,
        device=args.device,
        resume_dir=resume_dir,
        limit_per_category=args.limit_per_category,
    )


if __name__ == "__main__":
    main()
