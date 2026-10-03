from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .paths import project_root, runs_dir

_JOBS: dict[int, subprocess.Popen[str]] = {}
_LOCK = threading.Lock()


def _jobs_path() -> Path:
    return runs_dir() / "launcher-jobs.json"


def _load_records() -> list[dict[str, Any]]:
    path = _jobs_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_records(records: list[dict[str, Any]]) -> None:
    path = _jobs_path()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    temporary.replace(path)


def _record_finished_job(pid: int, exit_code: int) -> None:
    with _LOCK:
        records = _load_records()
        item = next((record for record in records if int(record.get("pid", -1)) == pid), None)
        if item is not None:
            item["exit_code"] = exit_code
            item["finished_at"] = time.time()
            item["status"] = (
                "stopped"
                if exit_code == 0 and item.get("stop_requested")
                else "complete"
                if exit_code == 0
                else "failed"
            )
            if exit_code < 0:
                item["termination_signal"] = -exit_code
            _save_records(records)
        _JOBS.pop(pid, None)


def _watch_job(process: subprocess.Popen[str]) -> None:
    _record_finished_job(process.pid, process.wait())


def _pid_is_training(pid: int) -> bool:
    command_path = Path(f"/proc/{pid}/cmdline")
    try:
        command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return any(
        marker in command
        for marker in (
            "exotic_trainer.cli train",
            "exotic-trainer train",
            "exotic_trainer.cli explore",
            "exotic-trainer explore",
            "exotic_trainer.cli sealed-eval",
            "exotic-trainer sealed-eval",
            "exotic_trainer.cli pair-pipeline",
            "exotic-trainer pair-pipeline",
            "exotic_trainer.cli dev-pair-pipeline",
            "exotic-trainer dev-pair-pipeline",
            "exotic_trainer.cli resume-dev-pair-pipeline",
            "exotic-trainer resume-dev-pair-pipeline",
            "exotic_trainer.cli routing-screening-pipeline",
            "exotic-trainer routing-screening-pipeline",
            "exotic_trainer.cli resume-routing-screening-pipeline",
            "exotic-trainer resume-routing-screening-pipeline",
            "exotic_trainer.cli geometry-screening-pipeline",
            "exotic-trainer geometry-screening-pipeline",
            "exotic_trainer.cli resume-geometry-screening-pipeline",
            "exotic-trainer resume-geometry-screening-pipeline",
            "exotic_trainer.cli ablation-pipeline",
            "exotic-trainer ablation-pipeline",
            "exotic_trainer.cli literal-lock-pipeline",
            "exotic-trainer literal-lock-pipeline",
            "exotic_trainer.cli resume-literal-lock-pipeline",
            "exotic-trainer resume-literal-lock-pipeline",
            "exotic_trainer.cli focused-human-pipeline",
            "exotic-trainer focused-human-pipeline",
            "exotic_trainer.cli resume-focused-human-pipeline",
            "exotic-trainer resume-focused-human-pipeline",
            "exotic_trainer.cli bfcl-setup",
            "exotic-trainer bfcl-setup",
            "exotic_trainer.cli bfcl-run",
            "exotic-trainer bfcl-run",
            "exotic_trainer.cli bfcl-v3-pair-pipeline",
            "exotic-trainer bfcl-v3-pair-pipeline",
        )
    )


def job_status() -> list[dict[str, Any]]:
    with _LOCK:
        records = _load_records()
        changed = False
        for item in records:
            pid = int(item["pid"])
            process = _JOBS.get(pid)
            if process is not None:
                code = process.poll()
                if code is None:
                    new_status = "stopping" if item.get("stop_requested") else "running"
                else:
                    new_status = (
                        "stopped"
                        if code == 0 and item.get("stop_requested")
                        else "complete"
                        if code == 0
                        else "failed"
                    )
                    item["exit_code"] = code
                    item["finished_at"] = time.time()
                    _JOBS.pop(pid, None)
            elif item.get("status") in {"running", "stopping"}:
                new_status = item["status"] if _pid_is_training(pid) else "ended"
            else:
                new_status = item.get("status", "unknown")
            if item.get("status") != new_status:
                item["status"] = new_status
                changed = True
        if changed:
            _save_records(records)
        return sorted(records, key=lambda item: float(item.get("started_at", 0)), reverse=True)


def clear_finished_jobs() -> list[dict[str, Any]]:
    """Clear finished launcher records from the GUI; log files remain on disk."""
    with _LOCK:
        active = [item for item in _load_records() if item.get("status") in {"running", "stopping"}]
        _save_records(active)
    return job_status()


def _start_work_job(arguments: list[str], kind: str, subject: str) -> dict[str, Any]:
    active = [item for item in job_status() if item["status"] in {"running", "stopping"}]
    if active:
        raise RuntimeError(
            f"training job {active[0]['pid']} is already {active[0]['status']}; stop it first"
        )
    stamp = f"{int(time.time())}-{os.getpid()}"
    log_path = runs_dir() / f"launcher-{stamp}.log"
    stop_path = runs_dir() / f"launcher-{stamp}.stop"
    log_handle = log_path.open("w", encoding="utf-8")
    environment = os.environ.copy()
    environment["EXOTIC_TRAINER_ROOT"] = str(project_root())
    environment["EXOTIC_TRAINER_STOP_FILE"] = str(stop_path)
    process = subprocess.Popen(
        [sys.executable, "-m", "exotic_trainer.cli", *arguments],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        cwd=project_root(),
        env=environment,
    )
    log_handle.close()
    item: dict[str, Any] = {
        "pid": process.pid,
        "kind": kind,
        "status": "running",
        "log": str(log_path),
        "subject": subject,
        "stop_file": str(stop_path),
        "stop_requested": False,
        "started_at": time.time(),
    }
    with _LOCK:
        records = _load_records()
        records.append(item)
        _save_records(records[-100:])
        _JOBS[process.pid] = process
    threading.Thread(target=_watch_job, args=(process,), daemon=True).start()
    return item


def start_training_job(recipe_path: Path) -> dict[str, Any]:
    return _start_work_job(["train", str(recipe_path)], "training", str(recipe_path))


def start_exploration_job(sheet_id: str) -> dict[str, Any]:
    return _start_work_job(["explore", sheet_id], "exploration", sheet_id)


def start_sealed_eval_job(
    run_id: str, suite_id: str, max_samples: int, compare_base: bool
) -> dict[str, Any]:
    arguments = [
        "sealed-eval",
        run_id,
        suite_id,
        "--max-samples",
        str(max_samples),
    ]
    if compare_base:
        arguments.append("--compare-base")
    return _start_work_job(arguments, "sealed-eval", f"{run_id}:{suite_id}")


def start_sealed_eval_batch_job(
    run_ids: list[str], suite_id: str, max_samples: int, compare_base: bool
) -> dict[str, Any]:
    arguments = ["sealed-eval-batch", suite_id, "--max-samples", str(max_samples)]
    for run_id in run_ids:
        arguments.extend(["--run-id", run_id])
    if compare_base:
        arguments.append("--compare-base")
    return _start_work_job(
        arguments,
        "sealed-eval-batch",
        f"{','.join(run_ids)}:{suite_id}",
    )


def start_pair_pipeline_job(
    baseline_sheet_id: str,
    exotic_sheet_id: str,
    suite_id: str,
    max_samples: int,
    compare_base: bool,
) -> dict[str, Any]:
    arguments = [
        "pair-pipeline",
        baseline_sheet_id,
        exotic_sheet_id,
        suite_id,
        "--max-samples",
        str(max_samples),
    ]
    if compare_base:
        arguments.append("--compare-base")
    return _start_work_job(
        arguments,
        "pair-pipeline",
        f"{baseline_sheet_id},{exotic_sheet_id}:{suite_id}",
    )


def start_dev_pair_pipeline_job(
    baseline_sheet_id: str,
    exotic_sheet_id: str,
    max_samples: int,
    compare_base: bool,
) -> dict[str, Any]:
    arguments = [
        "dev-pair-pipeline",
        baseline_sheet_id,
        exotic_sheet_id,
        "--max-samples",
        str(max_samples),
    ]
    if compare_base:
        arguments.append("--compare-base")
    return _start_work_job(
        arguments,
        "dev-pair-pipeline",
        f"{baseline_sheet_id},{exotic_sheet_id}:external-dev",
    )


def resume_dev_pair_pipeline_job(pipeline_id: str | None = None) -> dict[str, Any]:
    arguments = ["resume-dev-pair-pipeline"]
    if pipeline_id:
        arguments.extend(["--pipeline-id", pipeline_id])
    return _start_work_job(
        arguments,
        "resume-dev-pair-pipeline",
        pipeline_id or "latest-external-dev-pipeline",
    )


def start_routing_screening_job(
    sheet_ids: dict[str, str],
    max_samples: int,
    compare_base: bool,
) -> dict[str, Any]:
    arguments = [
        "routing-screening-pipeline",
        sheet_ids["A-baseline"],
        sheet_ids["B-pi-noise"],
        sheet_ids["C-role-ce"],
        sheet_ids["D-pi-role-ce"],
        "--max-samples",
        str(max_samples),
    ]
    if compare_base:
        arguments.append("--compare-base")
    return _start_work_job(
        arguments,
        "routing-screening-pipeline",
        ",".join(sheet_ids[label] for label in sorted(sheet_ids)),
    )


def resume_routing_screening_job(pipeline_id: str | None = None) -> dict[str, Any]:
    arguments = ["resume-routing-screening-pipeline"]
    if pipeline_id:
        arguments.extend(["--pipeline-id", pipeline_id])
    return _start_work_job(
        arguments,
        "resume-routing-screening-pipeline",
        pipeline_id or "latest-routing-screening-pipeline",
    )


def start_geometry_screening_job(
    baseline_run_id: str,
    noise_run_id: str,
    geometry_sheet_id: str,
    combined_sheet_id: str,
    source_pipeline_id: str,
    max_samples: int,
) -> dict[str, Any]:
    arguments = [
        "geometry-screening-pipeline",
        baseline_run_id,
        noise_run_id,
        geometry_sheet_id,
        combined_sheet_id,
        source_pipeline_id,
        "--max-samples",
        str(max_samples),
    ]
    return _start_work_job(
        arguments,
        "geometry-screening-pipeline",
        f"{baseline_run_id},{noise_run_id},{geometry_sheet_id},{combined_sheet_id}:external-dev",
    )


def resume_geometry_screening_job(pipeline_id: str | None = None) -> dict[str, Any]:
    arguments = ["resume-geometry-screening-pipeline"]
    if pipeline_id:
        arguments.extend(["--pipeline-id", pipeline_id])
    return _start_work_job(
        arguments,
        "resume-geometry-screening-pipeline",
        pipeline_id or "latest-geometry-screening-pipeline",
    )


def start_ablation_pipeline_job(
    baseline_run_id: str,
    noise_run_id: str,
    geometry_sheet_id: str,
    combined_sheet_id: str,
    suite_id: str,
    max_samples: int,
    compare_base: bool,
) -> dict[str, Any]:
    arguments = [
        "ablation-pipeline",
        baseline_run_id,
        noise_run_id,
        geometry_sheet_id,
        combined_sheet_id,
        suite_id,
        "--max-samples",
        str(max_samples),
    ]
    if compare_base:
        arguments.append("--compare-base")
    return _start_work_job(
        arguments,
        "ablation-pipeline",
        f"{baseline_run_id},{noise_run_id},{geometry_sheet_id},{combined_sheet_id}:{suite_id}",
    )


def start_literal_lock_pipeline_job(
    sheet_ids: dict[str, str],
    suite_id: str,
    dev_samples: int,
    sealed_samples: int,
    compare_base: bool,
) -> dict[str, Any]:
    labels = (
        "A-baseline",
        "B-pi-noise",
        "F-pi-geometry",
        "G-pi-literal-lock",
    )
    arguments = [
        "literal-lock-pipeline",
        *(sheet_ids[label] for label in labels),
        suite_id,
        "--dev-samples",
        str(dev_samples),
        "--sealed-samples",
        str(sealed_samples),
    ]
    if compare_base:
        arguments.append("--compare-base")
    return _start_work_job(
        arguments,
        "literal-lock-pipeline",
        f"{','.join(sheet_ids[label] for label in labels)}:{suite_id}",
    )


def resume_literal_lock_pipeline_job(pipeline_id: str | None = None) -> dict[str, Any]:
    arguments = ["resume-literal-lock-pipeline"]
    if pipeline_id:
        arguments.extend(["--pipeline-id", pipeline_id])
    return _start_work_job(
        arguments,
        "resume-literal-lock-pipeline",
        pipeline_id or "latest-literal-lock-pipeline",
    )


def start_focused_human_pipeline_job(samples: int = 520) -> dict[str, Any]:
    return _start_work_job(
        ["focused-human-pipeline", "--samples", str(int(samples))],
        "focused-human-pipeline",
        f"human+strict+elastic:pi,sqrt2,anchor-geometry,literal-lock:{int(samples)}",
    )


def resume_focused_human_pipeline_job(
    pipeline_id: str | None = None,
) -> dict[str, Any]:
    arguments = ["resume-focused-human-pipeline"]
    if pipeline_id:
        arguments.extend(["--pipeline-id", pipeline_id])
    return _start_work_job(
        arguments,
        "resume-focused-human-pipeline",
        pipeline_id or "latest-focused-human-pipeline",
    )


def start_bfcl_setup_job(version: str) -> dict[str, Any]:
    return _start_work_job(
        ["bfcl-setup", version],
        "bfcl-setup",
        f"BFCL {version} pinned environment",
    )


def start_bfcl_run_job(config_path: Path) -> dict[str, Any]:
    return _start_work_job(
        ["bfcl-run", str(config_path)],
        "bfcl-run",
        str(config_path),
    )


def start_bfcl_v3_pair_pipeline_job(steps: int = 400) -> dict[str, Any]:
    return _start_work_job(
        ["bfcl-v3-pair-pipeline", "--steps", str(int(steps))],
        "bfcl-v3-pair-pipeline",
        f"rsLoRA baseline vs natural pi decimal-pair noise; full BFCL v3; {int(steps)} steps",
    )


def stop_latest_training_job() -> dict[str, Any]:
    with _LOCK:
        records = _load_records()
        active = [item for item in records if item.get("status") in {"running", "stopping"}]
        active.sort(key=lambda item: float(item.get("started_at", 0)), reverse=True)
        if not active:
            return {"status": "idle", "message": "No active training job"}
        item = active[0]
        pid = int(item["pid"])
        if not _pid_is_training(pid):
            item["status"] = "ended"
            _save_records(records)
            return {"status": "ended", "pid": pid, "message": "Training process already ended"}
        stop_path = Path(item["stop_file"])
        stop_path.parent.mkdir(parents=True, exist_ok=True)
        stop_path.touch()
        item["stop_requested"] = True
        item["status"] = "stopping"
        item["stop_requested_at"] = time.time()
        _save_records(records)
        return {
            "status": "stopping",
            "pid": pid,
            "message": "Graceful stop requested; the current operation will finish safely.",
        }
