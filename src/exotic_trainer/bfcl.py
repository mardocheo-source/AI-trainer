from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from .paths import project_root, runs_dir
from .registry import Registry

BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla.git"
BFCL_SPECS: dict[str, dict[str, str]] = {
    # The commit that introduced the official BFCL V3 datasets and evaluator.
    "v3": {
        "commit": "70b6a4a2144597b1f99d1f4d3185d35d7ee532a4",
        "label": "BFCL v3 release (Liquid model-card compatible protocol)",
    },
    # Pinned official V4 tree.  Do not silently move this pin: benchmark protocol changes
    # must produce a different protocol id and a fresh result directory.
    "v4": {
        "commit": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        "label": "BFCL v4 official evaluator",
    },
}

BFCL_PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "label": "Smoke / diagnostic (partial, NOT leaderboard-publishable)",
        "categories": ["simple_python", "irrelevance"],
        "partial": True,
    },
    "non_live": {
        "label": "Official non-live section",
        "categories": ["non_live"],
        "partial": False,
    },
    "single_turn": {
        "label": "Official single-turn section",
        "categories": ["single_turn"],
        "partial": False,
    },
    "full": {
        "label": "Official full suite (requires optional heavy agentic extras)",
        "categories": ["all_scoring"],
        "partial": False,
    },
}

BFCL_COMPACT_DEPENDENCIES = (
    "openai>=1.86.0",
    "python-dotenv>=1.0.1",
    "tree_sitter==0.21.3",
    "tree-sitter-java==0.21.0",
    "tree-sitter-javascript==0.21.4",
    "tenacity>=8.5.0",
    "overrides>=7.7.0",
)
BFCL_REQUIRED_MODULES = (
    "openai",
    "dotenv",
    "tree_sitter",
    "tree_sitter_java",
    "tree_sitter_javascript",
    "tenacity",
    "overrides",
)

MIB = 1024 * 1024
GIB = 1024 * MIB
# Conservative upper bounds for result logs, rendered prompts and score artifacts.  These
# are deliberately larger than a typical run so a nearly-full training disk is never filled
# by surprise.  The compact runner does not support the dependency-heavy V4 agentic bundle.
BFCL_PROFILE_DISK_ESTIMATES = {
    "smoke": 64 * MIB,
    "non_live": 2 * GIB,
    "single_turn": 2 * GIB,
    "full": 6 * GIB,
}
BFCL_DISK_RESERVE = 512 * MIB
BFCL_REQUEST_TIMEOUT_SECONDS = 210.0
BFCL_GENERATION_TIMEOUT_SECONDS = 180.0
BFCL_LONG_PROMPT_CACHE_THRESHOLD = 4096
BFCL_XPU_MEMORY_FRACTION = 0.85
BFCL_MAX_NEW_TOKENS = 1024


def _bfcl_root() -> Path:
    root = project_root() / ".cache" / "bfcl"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _source_dir(version: str) -> Path:
    return _bfcl_root() / f"source-{version}"


def _benchmark_dir(version: str) -> Path:
    return _source_dir(version) / "berkeley-function-call-leaderboard"


def bfcl_environment_status(version: str | None = None) -> dict[str, Any]:
    versions = [version] if version else sorted(BFCL_SPECS)
    project_disk = shutil.disk_usage(project_root())
    output_disk = shutil.disk_usage(runs_dir())
    payload: dict[str, Any] = {
        "status": "ready",
        "versions": {},
        "project_disk": {
            "path": str(project_root()),
            "free_bytes": project_disk.free,
            "free_gib": round(project_disk.free / GIB, 3),
            "required_reserve_bytes": BFCL_DISK_RESERVE,
        },
        "output_disk": {
            "path": str(runs_dir()),
            "free_bytes": output_disk.free,
            "free_gib": round(output_disk.free / GIB, 3),
            "required_reserve_bytes": BFCL_DISK_RESERVE,
        },
        "compact_note": (
            "No duplicate Torch, CUDA, vLLM or vendor SDKs. Full V4 agentic sections are "
            "intentionally unavailable in compact mode."
        ),
    }
    for item in versions:
        if item not in BFCL_SPECS:
            raise ValueError(f"Unsupported BFCL version: {item}")
        source = _source_dir(item)
        commit = None
        if (source / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            commit = result.stdout.strip() if result.returncode == 0 else None
        ready = bool(
            commit == BFCL_SPECS[item]["commit"]
            and _benchmark_dir(item).is_dir()
            and all(importlib.util.find_spec(module) for module in BFCL_REQUIRED_MODULES)
        )
        payload["versions"][item] = {
            "ready": ready,
            "label": BFCL_SPECS[item]["label"],
            "expected_commit": BFCL_SPECS[item]["commit"],
            "installed_commit": commit,
            "source": str(source),
            "python": sys.executable,
            "install_mode": "compact-project-venv",
            "disk_bytes": sum(
                path.stat().st_size
                for path in _benchmark_dir(item).rglob("*")
                if path.is_file()
            )
            if _benchmark_dir(item).is_dir()
            else 0,
        }
        if not ready:
            payload["status"] = "setup-required"
    return payload


def _run_checked(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def setup_bfcl_environment(version: str) -> dict[str, Any]:
    if version not in BFCL_SPECS:
        raise ValueError(f"Unsupported BFCL version: {version}")
    source = _source_dir(version)
    expected = BFCL_SPECS[version]["commit"]

    free = shutil.disk_usage(project_root()).free
    if free < BFCL_DISK_RESERVE + 64 * MIB:
        raise RuntimeError(
            "BFCL compact setup refused: less than 576 MiB is free on the project disk. "
            "Free space first; no files were downloaded."
        )

    if not (source / ".git").exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        source.mkdir(parents=True, exist_ok=True)
        _run_checked(["git", "init", str(source)])
        _run_checked(["git", "-C", str(source), "sparse-checkout", "init", "--cone"])
        _run_checked(
            [
                "git",
                "-C",
                str(source),
                "sparse-checkout",
                "set",
                "berkeley-function-call-leaderboard",
            ]
        )
    remote = subprocess.run(
        ["git", "-C", str(source), "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        check=False,
    )
    if remote.returncode != 0:
        _run_checked(["git", "-C", str(source), "remote", "add", "origin", BFCL_REPOSITORY])
    elif remote.stdout.strip() != BFCL_REPOSITORY:
        _run_checked(["git", "-C", str(source), "remote", "set-url", "origin", BFCL_REPOSITORY])
    _run_checked(
        [
            "git",
            "-C",
            str(source),
            "fetch",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            expected,
        ]
    )
    _run_checked(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"])

    uv = shutil.which("uv")
    if uv:
        _run_checked(
            [
                uv,
                "pip",
                "install",
                "--no-cache",
                "--python",
                sys.executable,
                *BFCL_COMPACT_DEPENDENCIES,
            ]
        )
    else:
        _run_checked(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                *BFCL_COMPACT_DEPENDENCIES,
            ]
        )
    return bfcl_environment_status(version)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_gateway(url: str, process: subprocess.Popen[Any], timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Local model gateway exited with code {process.returncode}")
        try:
            base = url.rstrip("/").removesuffix("/v1")
            with urllib.request.urlopen(f"{base}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(2)
    raise TimeoutError(f"Local model gateway was not ready after {timeout:.0f}s: {last_error}")


def _resolve_target(target: str) -> tuple[str, str | None, str]:
    kind, _, identifier = str(target).partition(":")
    if kind == "model":
        return Registry().resolve_model(identifier), None, identifier
    if kind == "run":
        run = Registry().get_run(identifier)
        if not run or run.get("status") != "complete":
            raise ValueError(f"Completed run not found: {identifier}")
        recipe = json.loads(run.get("recipe_json") or "{}")
        model = Registry().resolve_model(str(recipe.get("model") or ""))
        return model, str(run["output_dir"]), identifier
    raise ValueError("Target must be model:<path-or-id> or run:<completed-run-id>")


def build_bfcl_config(
    *,
    version: str,
    profile: str,
    handler: str,
    transport: str,
    target: str,
    endpoint: str,
    api_model: str,
    smoke_samples: int,
    device: str = "xpu",
) -> dict[str, Any]:
    if version not in BFCL_SPECS:
        raise ValueError(f"Unsupported BFCL version: {version}")
    if profile not in BFCL_PROFILES:
        raise ValueError(f"Unsupported BFCL profile: {profile}")
    if handler not in {"liquid-lfm2", "generic-openai-tools"}:
        raise ValueError(f"Unsupported BFCL handler: {handler}")
    if transport not in {"managed-local-xpu", "existing-openai-endpoint"}:
        raise ValueError(f"Unsupported BFCL transport: {transport}")
    if version == "v3" and profile == "smoke":
        raise ValueError("BFCL v3 release does not support honest partial scoring; choose non_live or full")
    if version == "v4" and profile == "full":
        raise ValueError(
            "Compact BFCL intentionally excludes the heavy agentic memory/web extras. "
            "Choose single_turn for the complete official core suite."
        )
    categories = list(BFCL_PROFILES[profile]["categories"])
    if version == "v3" and profile == "full":
        categories = ["all"]
    estimated_output_bytes = BFCL_PROFILE_DISK_ESTIMATES[profile]
    output_root = runs_dir()
    free_bytes = shutil.disk_usage(output_root).free
    required_bytes = estimated_output_bytes + BFCL_DISK_RESERVE
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"BFCL {profile} refused before launch: approximately "
            f"{estimated_output_bytes / GIB:.2f} GiB plus a 0.50 GiB safety reserve is "
            f"required, but only {free_bytes / GIB:.2f} GiB is free on "
            f"{output_root}. Choose a smaller scope or another runs directory first."
        )
    model_path = adapter_path = target_name = None
    if transport == "managed-local-xpu":
        model_path, adapter_path, target_name = _resolve_target(target)
    elif not endpoint.strip():
        raise ValueError("An OpenAI-compatible endpoint is required")
    return {
        "schema_version": 1,
        "version": version,
        "bfcl_commit": BFCL_SPECS[version]["commit"],
        "profile": profile,
        "categories": categories,
        "partial": bool(BFCL_PROFILES[profile]["partial"]),
        "handler": handler,
        "handler_provenance": (
            "local Liquid LFM2 compatibility adapter aligned with the proposed Gorilla "
            "BFCL handler c2e418a8f41b3d6a0871fb01d98a262714155a67 (not merged)"
            if handler == "liquid-lfm2"
            else "official BFCL OpenAICompletionsHandler"
        ),
        "transport": transport,
        "target": target,
        "target_name": target_name or api_model,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "endpoint": endpoint.rstrip("/") or None,
        "api_model": api_model.strip() or "local-model",
        "smoke_samples": max(2, int(smoke_samples)),
        "device": device,
        "estimated_output_bytes": estimated_output_bytes,
        "disk_free_bytes_at_configuration": free_bytes,
        "runs_dir": str(output_root),
        "disk_reserve_bytes": BFCL_DISK_RESERVE,
        "runtime_limits": {
            "request_timeout_seconds": BFCL_REQUEST_TIMEOUT_SECONDS,
            "generation_timeout_seconds": BFCL_GENERATION_TIMEOUT_SECONDS,
            "openai_max_retries": 0,
            "long_prompt_cache_threshold": BFCL_LONG_PROMPT_CACHE_THRESHOLD,
            "xpu_memory_fraction": BFCL_XPU_MEMORY_FRACTION,
            "max_new_tokens": BFCL_MAX_NEW_TOKENS,
        },
    }


def save_bfcl_config(config: dict[str, Any]) -> Path:
    identifier = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    output = runs_dir() / "bfcl" / identifier
    output.mkdir(parents=True, exist_ok=False)
    config = {**config, "bfcl_run_id": identifier, "output_dir": str(output)}
    path = output / "config.json"
    _atomic_json(path, config)
    _atomic_json(
        output / "state.json",
        {
            "bfcl_run_id": identifier,
            "status": "queued",
            "stage": "queued",
            "created_at": time.time(),
            "config": config,
        },
    )
    return path


def seed_bfcl_results(config_path: Path, source_output_dir: Path) -> dict[str, Any]:
    """Reuse valid official JSONL rows while making inference failures retryable.

    The BFCL v3 collector already skips IDs found in its result directory. Error rows
    must therefore be omitted from the recovered copy; otherwise an interrupted timeout
    would be treated as a completed test case forever.
    """

    config = json.loads(config_path.read_text(encoding="utf-8"))
    destination = Path(config["output_dir"]) / "result"
    source = Path(source_output_dir) / "result"
    if not source.is_dir():
        return {"status": "no-source", "source": str(source), "retained": 0, "retry": 0}
    destination.mkdir(parents=True, exist_ok=True)
    retained = retry = files = 0
    for source_path in sorted(source.rglob("*.json")):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        good_lines: list[str] = []
        bad_in_file = 0
        with source_path.open(encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                try:
                    item = json.loads(raw_line)
                except json.JSONDecodeError:
                    bad_in_file += 1
                    continue
                result = item.get("result")
                is_error = isinstance(result, str) and result.startswith("Error during inference:")
                if is_error:
                    bad_in_file += 1
                else:
                    good_lines.append(json.dumps(item, ensure_ascii=False))
        if bad_in_file == 0:
            try:
                os.link(source_path, destination_path)
            except OSError:
                shutil.copy2(source_path, destination_path)
        else:
            destination_path.write_text(
                "".join(f"{line}\n" for line in good_lines), encoding="utf-8"
            )
        files += 1
        retained += len(good_lines)
        retry += bad_in_file
    recovery = {
        "status": "seeded",
        "source": str(source),
        "destination": str(destination),
        "files": files,
        "retained": retained,
        "retry": retry,
        "copy_policy": "hard-link error-free files; rewrite files containing failed rows",
    }
    _atomic_json(Path(config["output_dir"]) / "recovery.json", recovery)
    return recovery


def _read_score_summary(score_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"csv_files": []}
    for path in sorted(score_dir.rglob("*.csv")):
        rows: list[dict[str, str]] = []
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            continue
        summary["csv_files"].append({"path": str(path), "rows": rows})
        if path.name == "data_overall.csv":
            summary["overall"] = rows
    return summary


def _generated_case_count(result_dir: Path) -> int:
    count = 0
    for path in result_dir.rglob("*.json") if result_dir.is_dir() else ():
        try:
            with path.open(encoding="utf-8") as handle:
                count += sum(1 for line in handle if line.strip())
        except OSError:
            continue
    return count


def xpu_vram_status() -> dict[str, Any] | None:
    """Read the small local Intel VRAM monitor without importing/initialising Torch."""

    candidates = [Path("/run/xe-gpu-vram")]
    candidates.extend(sorted(Path("/run").glob("xe-gpu-vram-*")))
    for path in candidates:
        try:
            fields = path.read_text(encoding="utf-8").split()
            used, total = int(fields[0]), int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
        return {
            "path": str(path),
            "used_bytes": used,
            "total_bytes": total,
            "used_gib": round(used / GIB, 3),
            "total_gib": round(total / GIB, 3),
            "fraction": used / total if total else 0.0,
        }
    return None


def _ensure_xpu_headroom(min_free_gib: float = 2.5) -> dict[str, Any] | None:
    status = xpu_vram_status()
    if status:
        free_gib = status["total_gib"] - status["used_gib"]
        if free_gib < min_free_gib:
            raise RuntimeError(
                "BFCL local gateway refused to load another model: Intel VRAM has only "
                f"{free_gib:.2f} GiB free ({status['used_gib']:.2f}/{status['total_gib']:.2f} GiB used). "
                "Stop the other XPU process first; CPU fallback is disabled."
            )
    return status


def _wait_for_xpu_release(timeout: float = 30.0, min_free_gib: float = 2.5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = xpu_vram_status()
        if status is None or (status["total_gib"] - status["used_gib"]) >= min_free_gib:
            return
        time.sleep(1)


def run_bfcl(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = Path(config["output_dir"])
    state_path = output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))

    def update(stage: str, **values: Any) -> None:
        state.update({"status": "running", "stage": stage, "updated_at": time.time(), **values})
        _atomic_json(state_path, state)

    environment_status = bfcl_environment_status(str(config["version"]))
    if environment_status["status"] != "ready":
        raise RuntimeError(
            f"BFCL {config['version']} environment is not installed; run BFCL setup first"
        )

    gateway: subprocess.Popen[Any] | None = None
    gateway_log = None
    try:
        endpoint = str(config.get("endpoint") or "")
        if config["transport"] == "managed-local-xpu":
            update("checking-xpu-headroom", xpu_vram_before_load=_ensure_xpu_headroom())
            port = _free_port()
            endpoint = f"http://127.0.0.1:{port}/v1"
            update("loading-local-model", endpoint=endpoint)
            gateway_log = (output / "gateway.log").open("w", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "exotic_trainer.cli",
                "serve",
                "--model",
                str(config["model_path"]),
                "--device",
                str(config.get("device") or "xpu"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--handler",
                str(config["handler"]),
                "--generation-timeout-seconds",
                str(
                    (config.get("runtime_limits") or {}).get(
                        "generation_timeout_seconds", BFCL_GENERATION_TIMEOUT_SECONDS
                    )
                ),
                "--long-prompt-cache-threshold",
                str(
                    (config.get("runtime_limits") or {}).get(
                        "long_prompt_cache_threshold", BFCL_LONG_PROMPT_CACHE_THRESHOLD
                    )
                ),
                "--xpu-memory-fraction",
                str(
                    (config.get("runtime_limits") or {}).get(
                        "xpu_memory_fraction", BFCL_XPU_MEMORY_FRACTION
                    )
                ),
                "--default-max-new-tokens",
                str(
                    (config.get("runtime_limits") or {}).get(
                        "max_new_tokens", BFCL_MAX_NEW_TOKENS
                    )
                ),
            ]
            if config.get("adapter_path"):
                command.extend(["--adapter", str(config["adapter_path"])])
            gateway = subprocess.Popen(
                command,
                cwd=project_root(),
                stdout=gateway_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_for_gateway(endpoint, gateway)

        update("bfcl-generation", endpoint=endpoint)
        runner = project_root() / "integrations" / "bfcl" / "official_runner.py"
        command = [
            sys.executable,
            str(runner),
            str(config_path),
            "--endpoint",
            endpoint,
            "--source",
            str(_benchmark_dir(str(config["version"]))),
        ]
        bfcl_log = (output / "bfcl.log").open("w", encoding="utf-8")
        try:
            started = time.monotonic()
            completed = subprocess.Popen(
                command,
                cwd=project_root(),
                stdout=bfcl_log,
                stderr=subprocess.STDOUT,
            )
            while completed.poll() is None:
                progress_path = output / "official-progress.json"
                progress: dict[str, Any] = {}
                if progress_path.is_file():
                    try:
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        progress = {}
                generated = _generated_case_count(output / "result")
                total = int(progress.get("total_cases") or 0)
                elapsed = time.monotonic() - started
                fraction = min(1.0, generated / total) if total else 0.0
                eta = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
                update(
                    str(progress.get("stage") or "bfcl-generation"),
                    endpoint=endpoint,
                    generated_cases=generated,
                    total_cases=total or None,
                    progress_percent=round(100.0 * fraction, 2),
                    elapsed_seconds=round(elapsed, 1),
                    estimated_remaining_seconds=round(eta, 1) if eta is not None else None,
                )
                time.sleep(5)
        finally:
            bfcl_log.close()
        if completed.returncode != 0:
            raise RuntimeError(
                f"BFCL runner failed with code {completed.returncode}; see {output / 'bfcl.log'}"
            )
        runner_result = json.loads((output / "official-result.json").read_text(encoding="utf-8"))
        score_summary = _read_score_summary(output / "score")
        state.update(
            {
                "status": "complete",
                "stage": "complete",
                "updated_at": time.time(),
                "finished_at": time.time(),
                "official_result": runner_result,
                "score_summary": score_summary,
                "publishable": bool(not config["partial"] and runner_result.get("complete")),
            }
        )
        _atomic_json(state_path, state)
        return state
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "stage": "failed",
                "updated_at": time.time(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _atomic_json(state_path, state)
        raise
    finally:
        if gateway is not None and gateway.poll() is None:
            try:
                os.killpg(gateway.pid, signal.SIGTERM)
                gateway.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if gateway.poll() is None:
                    os.killpg(gateway.pid, signal.SIGKILL)
            _wait_for_xpu_release()
        if gateway_log is not None:
            gateway_log.close()


def score_existing_bfcl(
    config_path: Path, *, categories: list[str] | None = None
) -> dict[str, Any]:
    """Score a complete BFCL result directory without loading or querying the model.

    This is deliberately separate from :func:`run_bfcl`: recovery never starts the local
    gateway, never allocates XPU memory and never regenerates a prediction.  It is safe to
    retry after a third-party executable-ground-truth API interrupts the official v3 scorer.
    """

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("version") != "v3":
        raise ValueError("score-only recovery is currently implemented for BFCL v3")
    output = Path(config["output_dir"])
    state_path = output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected = int(state.get("total_cases") or 0)
    generated = _generated_case_count(output / "result")
    if expected <= 0 or generated != expected:
        raise RuntimeError(
            f"cannot score incomplete BFCL predictions: {generated}/{expected or '?'}"
        )

    state.update(
        {
            "status": "running",
            "stage": "bfcl-score-only-recovery",
            "updated_at": time.time(),
            "generated_cases": generated,
            "progress_percent": 100.0,
            "error": None,
        }
    )
    _atomic_json(state_path, state)

    runner = project_root() / "integrations" / "bfcl" / "official_runner.py"
    command = [
        sys.executable,
        str(runner),
        str(config_path),
        "--source",
        str(_benchmark_dir("v3")),
        "--score-only",
    ]
    for category in categories or ():
        command.extend(["--score-category", str(category)])
    log_path = output / "bfcl-score-recovery.log"
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n--- score-only recovery {time.time()} ---\n")
            completed = subprocess.run(
                command,
                cwd=project_root(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"BFCL score-only runner failed with code {completed.returncode}; "
                f"see {log_path}"
            )
        runner_result = json.loads(
            (output / "official-result.json").read_text(encoding="utf-8")
        )
        complete_suite = not categories
        state.update(
            {
                "status": "complete" if complete_suite else "partially-scored",
                "stage": "complete" if complete_suite else "partial-score-complete",
                "updated_at": time.time(),
                "finished_at": time.time() if complete_suite else state.get("finished_at"),
                "official_result": runner_result,
                "score_summary": _read_score_summary(output / "score"),
                "publishable": bool(complete_suite and runner_result.get("complete")),
                "score_recovery": {
                    "mode": "score-only",
                    "categories": categories or ["all configured categories"],
                    "predictions_reused": generated,
                    "log": str(log_path),
                },
            }
        )
        _atomic_json(state_path, state)
        return state
    except Exception as error:
        has_partial_scores = bool((state.get("score_summary") or {}).get("csv_files"))
        state.update(
            {
                "status": "partially-scored" if has_partial_scores else "failed",
                "stage": (
                    "score-only-blocked-with-partial-results"
                    if has_partial_scores
                    else "score-only-failed"
                ),
                "updated_at": time.time(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _atomic_json(state_path, state)
        raise


def list_bfcl_runs() -> list[dict[str, Any]]:
    root = runs_dir() / "bfcl"
    if not root.exists():
        return []
    items = []
    for path in root.glob("*/state.json"):
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(items, key=lambda item: float(item.get("created_at", 0)), reverse=True)
