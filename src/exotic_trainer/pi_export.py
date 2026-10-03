from __future__ import annotations

import json
from pathlib import Path


def write_pi_models(model_id: str, endpoint: str, output: Path) -> Path:
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "providers": {
            "exotic-local": {
                "baseUrl": endpoint.rstrip("/"),
                "api": "openai-completions",
                "apiKey": "local",
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsUsageInStreaming": False,
                    "maxTokensField": "max_tokens",
                },
                "models": [
                    {
                        "id": model_id,
                        "name": f"{model_id} (local fine-tune)",
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": 32768,
                        "maxTokens": 8192,
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                    }
                ],
            }
        }
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination

