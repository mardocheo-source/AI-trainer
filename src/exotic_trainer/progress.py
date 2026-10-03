from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class EventWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **payload: Any) -> None:
        record = {"time": time.time(), "event": event, **payload}
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_events(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    output = []
    for line in lines:
        try:
            output.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return output

