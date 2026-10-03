"""Load local BFCL credentials without displaying or inventing secret values."""

from __future__ import annotations

import os
from pathlib import Path

from .paths import project_root

BFCL_KEYS = ("RAPID_API_KEY", "GEOCODE_API_KEY", "EXCHANGERATE_API_KEY", "OMDB_API_KEY")


def load_bfcl_environment(*, require_live: bool = False) -> None:
    """Load an explicit file or .secrets/bfcl.env; the shell takes precedence.

    Empty defaults accommodate BFCL v3's check for an empty string rather than
    None. Local model authentication is separate from these external API keys.
    No requests are made and no values are logged by this function.
    """
    from dotenv import dotenv_values

    configured = os.environ.get("AI_TRAINER_ENV_FILE")
    path = Path(configured).expanduser() if configured else project_root() / ".secrets/bfcl.env"
    if not path.is_absolute():
        path = project_root() / path
    if configured and not path.is_file():
        raise FileNotFoundError("AI_TRAINER_ENV_FILE must identify an existing local file")
    if path.is_file():
        values = dotenv_values(path, interpolate=False)
        for name in BFCL_KEYS:
            if name in values:
                os.environ.setdefault(name, values[name] or "")
    for name in BFCL_KEYS:
        os.environ.setdefault(name, "")
    if require_live:
        invalid = []
        for name in BFCL_KEYS:
            value = os.environ[name].strip()
            lower = value.lower()
            if (
                not value
                or lower in {"empty", "local", "changeme"}
                or any(
                    marker in lower
                    for marker in (
                        "placeholder",
                        "example",
                        "dummy",
                        "fake",
                        "your_key",
                        "your-key",
                    )
                )
            ):
                invalid.append(name)
        if invalid:
            raise ValueError(
                "Live BFCL execution requires configured credentials: " + ", ".join(invalid)
            )
