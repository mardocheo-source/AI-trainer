import json

from exotic_trainer import jobs


def test_finished_job_persists_negative_exit_signal(tmp_path, monkeypatch) -> None:
    path = tmp_path / "jobs.json"
    monkeypatch.setattr(jobs, "_jobs_path", lambda: path)
    path.write_text(
        json.dumps([{"pid": 42, "status": "running", "stop_requested": False}]),
        encoding="utf-8",
    )

    jobs._record_finished_job(42, -9)

    record = json.loads(path.read_text(encoding="utf-8"))[0]
    assert record["status"] == "failed"
    assert record["exit_code"] == -9
    assert record["termination_signal"] == 9


def test_clear_finished_jobs_preserves_active_records(tmp_path, monkeypatch) -> None:
    path = tmp_path / "jobs.json"
    monkeypatch.setattr(jobs, "_jobs_path", lambda: path)
    path.write_text(
        json.dumps(
            [
                {"pid": 1, "status": "complete", "started_at": 1},
                {"pid": 2, "status": "running", "started_at": 2},
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(jobs, "_pid_is_training", lambda pid: pid == 2)

    remaining = jobs.clear_finished_jobs()

    assert [item["pid"] for item in remaining] == [2]
