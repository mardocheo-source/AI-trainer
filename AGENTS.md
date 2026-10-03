# Instructions for AI agents

Work only in this AI-trainer repository. The original AI-trainier repository is a private archive.

## Credentials and private data
- Real BFCL keys belong in the ignored local file `.secrets/bfcl.env`, or an explicit file selected by `AI_TRAINER_ENV_FILE`.
- `.env.example` contains intentionally empty values. Do not invent real keys or fill this tracked template.
- Never read or print secret values in terminal output, chat, logs, reports, commits, test snapshots or prompts. Check names/presence only.
- Never add ignored secret files with `git add -f`. Do not follow links or copy credentials into tracked files.
- Shell variables take precedence over the local credential file. Offline tests use synthetic fixtures and must not call external APIs.
- If credentials appear in a tracked file or history, stop publication, report only path/category, and require rotation and history review.
- Do not publish local datasets, models, checkpoints, conversation logs or private audit reports.

## Verification
Use Python 3.12. Run `python -m pytest tests` before committing behavior changes.
Use small smoke tests for training; do not start long training runs or download large models implicitly.
Bound test processes, RAM and threads. Record limitations honestly; passing unit tests is not an end-to-end model benchmark.
