# AI-trainer

A Python research toolkit for parameter-efficient training and evaluation of agentic language models, with support for Intel XPU and CPU diagnostics.

**Research use only.** This is an experimental research project intended for exploration and evaluation. It is not intended for real-world deployment or production use.

This repository is the public-source edition of a private research project. It includes the application code, an isolated BFCL adapter, small synthetic examples and automated tests. Private datasets, model weights, checkpoints, unbundled historical campaigns, experiment reports and the original Git history are intentionally excluded. Unit tests do not establish model quality or reproduce past research scores.

## Origin and recorded development history

This public-source edition is a **curated, sanitised snapshot of earlier research development**, derived from the larger and more complex private [AI-trainier research workspace](https://github.com/mardocheo-source/AI-trainier). The original source, historical campaigns and research artifacts remain private on GitHub. The short public Git history records the clean export and privacy work, not the start of the project: this public edition began on **3 October 2026**.

The original recorded Git history spans **10 August–3 October 2026**, with **26 distinct reachable commits** across all local refs and current GitHub branches/tags, checked on 4 October 2026. Dates use the original UTC+09:00 timezone; commits shared by branches are counted once.

| Recorded month | Original commits |
| --- | ---: |
| August 2026 | 13 |
| September 2026 | 11 |
| October 2026, through the 3rd | 2 |
| **Total** | **26** |

Selected milestones from the private history (short hashes identify private commits, not commits in this public export):

- **10 August:** initial research-application/package history — `eb54d04144cf`.
- **16 August:** BFCL integration in the original source — `6b360c829cc1`.
- **25 August–12 September:** experimental campaign infrastructure and subsequent revisions — `cf6026fbbd97` → `712d136bcbb1`; those full campaigns are not bundled here.

The exported source includes reviewed working-tree changes as well as committed work. These counts describe the recorded history, not every experiment or time spent. The original development history was deliberately excluded from the public export. This remains **scientific research and experimental software only, not intended for real-world deployment**.

## Installation

Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
exotic-trainer --help
```

For the full test suite and model training, install a compatible PyTorch build for your hardware, then `python -m pip install -e '.[train]'` and run `python -m pytest tests`. For CPU-only checks, install PyTorch using `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` first. PyTorch is selected separately because CPU, CUDA and Intel XPU installations differ. Models must be obtained independently under their own licenses. The example recipe expects a local model at `models/LFM2.5-1.2B-Instruct`; edit this path for your setup.

## Local credentials

Most local tooling and the test suite do not need external API keys. Live BFCL execution uses four provider keys.

```bash
mkdir -p .secrets
cp .env.example .secrets/bfcl.env
chmod 700 .secrets
chmod 600 .secrets/bfcl.env
```

Edit only the local `.secrets/bfcl.env`. It is ignored by Git. Never fill or commit `.env.example`, which deliberately contains empty values. You may select another local file with `AI_TRAINER_ENV_FILE` (relative paths are resolved from the project root); environment variables already set in your shell take precedence. Variables in credential values are not expanded. Missing or obvious placeholder keys block live BFCL execution with an error containing variable names only. These checks do not prove provider validity.

The BFCL v3 adapter no longer auto-loads credentials from the downloaded evaluator's `.env`. Move that local configuration into the location documented above. Its executable/REST categories can invoke external APIs and incur charges; the synthetic test suite does not make those calls.

## What is included

- Dataset normalization, secret-pattern screening, recipes and a local SQLite registry.
- LoRA training infrastructure and experimental regularization/data-generation modules.
- Local serving and agent/tool evaluation helpers.
- Pinned official BFCL integration; evaluator source and datasets are downloaded separately.
- Unit tests with synthetic fixtures instead of private benchmark exports.

`recipes/smoke.yaml` and `examples/tiny-agent.jsonl` illustrate configuration. They are not a useful training corpus or a benchmark result. GPU training, provider API calls and full BFCL runs require separately provisioned resources and are not guaranteed by a passing unit suite.

## Verified checks

The publication preparation passed 99 automated tests in a fresh Python 3.12 environment, including a real one-step CPU LoRA run on a randomly initialized 6,800-parameter model with synthetic data. An adapter was saved. This validates a small execution path; XPU training, full campaigns and live BFCL/provider calls were not exercised.

## Development and provenance

Develop in this repository on `main`; the original `AI-trainier` remains a private archive. Do not develop in both folders. See [agent instructions](AGENTS.md), [security guidance](docs/SECURITY.md) and [third-party sources](docs/THIRD_PARTY.md).

The project metadata declares the source license as MIT. Model, dataset and dependency licenses apply separately.
