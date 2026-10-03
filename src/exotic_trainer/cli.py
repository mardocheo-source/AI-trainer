from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .dataset_probe import inspect_dataset
from .doctor import system_report
from .model_probe import choose_lora_targets, inspect_model
from .recipe import default_recipe, load_recipe, save_recipe
from .registry import Registry

app = typer.Typer(help="Train small agentic coding LLMs on Intel XPU.", no_args_is_help=True)
model_app = typer.Typer(help="Inspect and register local models.", no_args_is_help=True)
dataset_app = typer.Typer(
    help="Inspect and register local or Hugging Face datasets.", no_args_is_help=True
)
recipe_app = typer.Typer(help="Create and validate training recipes.", no_args_is_help=True)
app.add_typer(model_app, name="model")
app.add_typer(dataset_app, name="dataset")
app.add_typer(recipe_app, name="recipe")
console = Console()
DEFAULT_PI_OUTPUT = Path("artifacts/pi-models.json")


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json")) -> None:
    """Check the venv, packages and Intel XPU runtime."""
    report = system_report()
    if json_output:
        console.print_json(data=report)
        return
    console.print(f"Python: {report['python']}  venv: {'yes' if report['venv'] else 'NO'}")
    console.print(f"Platform: {report['platform']}")
    console.print(f"Intel XPU available: {report['torch_xpu_available']}")
    for device in report["xpu_devices"]:
        console.print(f"  XPU {device['index']}: {device['name']}")
    table = Table("Package", "Version")
    for name, version in report["packages"].items():
        table.add_row(name, version or "not installed")
    console.print(table)
    for error in report["errors"]:
        console.print(f"[yellow]{error}[/yellow]")


@model_app.command("add")
def model_add(path: str, name: str | None = None) -> None:
    probe = inspect_model(path)
    if name:
        probe.name = name
    identifier = Registry().upsert_model(probe)
    console.print(f"[green]Registered model #{identifier}[/green] {probe.name}")
    console.print(
        f"{probe.parameter_count / 1e9:.3f}B parameters, {probe.dtype}, "
        f"{probe.disk_bytes / 2**30:.2f} GiB"
    )
    targets = choose_lora_targets(probe, "auto")
    console.print(f"Auto LoRA targets: {len(targets)} modules")


@model_app.command("inspect")
def model_inspect(path: str, json_output: bool = typer.Option(False, "--json")) -> None:
    probe = inspect_model(path)
    if json_output:
        console.print_json(probe.model_dump_json())
    else:
        console.print(probe.model_dump())
        console.print({"auto_lora_targets": choose_lora_targets(probe, "auto")})


@model_app.command("list")
def model_list() -> None:
    table = Table("ID", "Name", "Architecture", "Parameters", "Path")
    for item in Registry().list_models():
        metadata = item["metadata"]
        table.add_row(
            str(item["id"]),
            item["name"],
            metadata["architecture"],
            f"{metadata['parameter_count'] / 1e9:.3f}B",
            item["path"],
        )
    console.print(table)


@dataset_app.command("add")
def dataset_add(source: str, name: str | None = None) -> None:
    probe = inspect_dataset(source, name)
    identifier = Registry().upsert_dataset(probe)
    console.print(f"[green]Registered dataset #{identifier}[/green] {probe.name}")
    if probe.secret_hits:
        console.print(f"[red]Warning: {probe.secret_hits} possible secret(s) in sampled rows[/red]")


@dataset_app.command("inspect")
def dataset_inspect(source: str) -> None:
    console.print_json(inspect_dataset(source).model_dump_json())


@dataset_app.command("list")
def dataset_list() -> None:
    table = Table("ID", "Name", "Type", "Rows", "Secrets", "Source")
    for item in Registry().list_datasets():
        metadata = item["metadata"]
        table.add_row(
            str(item["id"]),
            item["name"],
            metadata["source_type"],
            str(metadata.get("row_count") or "?"),
            str(metadata.get("secret_hits", 0)),
            item["source"],
        )
    console.print(table)


@dataset_app.command("fetch")
def dataset_fetch(
    source: str,
    destination: str | None = typer.Option(None, "--destination", "-d"),
    name: str | None = typer.Option(None, "--name"),
) -> None:
    """Download a Hugging Face dataset as inert local Parquet files and register it."""
    from .downloads import fetch_huggingface_dataset

    path = fetch_huggingface_dataset(source, destination)
    probe = inspect_dataset(str(path), name=name)
    identifier = Registry().upsert_dataset(probe)
    console.print(f"[green]Downloaded and registered dataset #{identifier}:[/green] {path}")


@dataset_app.command("validate")
def dataset_validate(source: str, rows: int = typer.Option(5, min=1, max=100)) -> None:
    """Load a small bounded sample and verify that it normalizes for training."""
    from .data_loading import validate_training_source

    resolved = Registry().resolve_dataset(source)
    console.print_json(
        data=validate_training_source(
            resolved, rows=rows, allowed_tools=["read", "write", "edit", "bash"]
        )
    )


@recipe_app.command("init")
def recipe_init(
    path: str,
    model: str = typer.Option("/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct", "--model"),
    dataset: list[str] | None = typer.Option(None, "--dataset"),
) -> None:
    datasets = dataset or ["CHANGE_ME.jsonl"]
    destination = save_recipe(default_recipe(model, datasets), path)
    console.print(f"[green]Created[/green] {destination}")


@recipe_app.command("validate")
def recipe_validate(path: str) -> None:
    recipe = load_recipe(path)
    console.print_json(data=recipe.model_dump(mode="json"))


@app.command()
def train(
    recipe_path: str,
    dry_run: bool = typer.Option(False, help="Resolve and validate without loading weights."),
) -> None:
    """Start a bounded rsLoRA training run."""
    from .trainer import run_training

    result = run_training(load_recipe(recipe_path), dry_run=dry_run)
    console.print_json(data=result)


@app.command()
def explore(sheet_id: str) -> None:
    """Run a saved workbook sheet's bounded LoRA/exotic exploration."""
    from .exploration import run_exploration

    console.print_json(data=run_exploration(sheet_id))


@app.command("sealed-eval")
def sealed_eval(
    run_id: str,
    suite_id: str,
    max_samples: int = typer.Option(32, min=1, max=1000),
    compare_base: bool = typer.Option(False, "--compare-base"),
) -> None:
    """Evaluate an adapter on a registered never-train sealed suite."""
    from .sealed import run_sealed_evaluation

    console.print_json(
        data=run_sealed_evaluation(
            run_id=run_id,
            suite_id=suite_id,
            max_samples=max_samples,
            compare_base=compare_base,
        )
    )


@app.command("sealed-eval-variant", hidden=True)
def sealed_eval_variant(
    run_id: str,
    suite_id: str,
    max_samples: int,
    variant: str,
    result_path: Path,
) -> None:
    """Internal isolated XPU worker for one sealed-evaluation variant."""
    from .sealed import run_sealed_variant

    run_sealed_variant(run_id, suite_id, max_samples, variant, result_path)


@app.command("sealed-eval-batch")
def sealed_eval_batch(
    suite_id: str,
    run_id: list[str] = typer.Option(..., "--run-id"),
    max_samples: int = typer.Option(32, min=1, max=1000),
    compare_base: bool = typer.Option(False, "--compare-base"),
    quiet: bool = typer.Option(
        False, "--quiet", help="Write result files without printing predictions."
    ),
) -> None:
    """Evaluate several adapters against one identical sealed protocol."""
    from .sealed import run_sealed_batch

    result = run_sealed_batch(
        run_ids=run_id,
        suite_id=suite_id,
        max_samples=max_samples,
        compare_base=compare_base,
    )
    if not quiet:
        console.print_json(data=result)
    else:
        console.print(
            {
                "status": result.get("status"),
                "suite_id": result.get("suite_id"),
                "runs_evaluated": result.get("runs_evaluated"),
                "result_paths": [item.get("result_path") for item in result.get("results", [])],
            }
        )


@app.command("prepare-human-agentic-ood")
def prepare_human_agentic_ood() -> None:
    """Build, audit and permanently register the novel-template final OOD suite."""
    from .human_agentic_ood import ensure_human_agentic_ood_suite

    console.print_json(data=ensure_human_agentic_ood_suite())


@app.command("dev-eval-variant", hidden=True)
def dev_eval_variant(
    run_id: str,
    source: str,
    max_samples: int,
    variant: str,
    result_path: Path,
) -> None:
    """Internal isolated XPU worker for one external-DEV variant."""
    from .dev_eval import run_dev_variant

    run_dev_variant(run_id, source, max_samples, variant, result_path)


@app.command("dev-pair-pipeline")
def dev_pair_pipeline(
    baseline_sheet_id: str,
    exotic_sheet_id: str,
    max_samples: int = typer.Option(520, min=1, max=5000),
    compare_base: bool = typer.Option(False, "--compare-base"),
) -> None:
    """Train a fair pair and evaluate both on the shared external DEV set."""
    from .pipeline import run_dev_pair_pipeline

    console.print_json(
        data=run_dev_pair_pipeline(
            baseline_sheet_id=baseline_sheet_id,
            exotic_sheet_id=exotic_sheet_id,
            max_samples=max_samples,
            compare_base=compare_base,
        )
    )


@app.command("resume-dev-pair-pipeline")
def resume_dev_pair_pipeline_command(
    pipeline_id: str | None = typer.Option(None, "--pipeline-id"),
) -> None:
    """Resume only the external-DEV evaluation of a completed fair pair."""
    from .pipeline import resume_dev_pair_pipeline

    console.print_json(data=resume_dev_pair_pipeline(pipeline_id))


@app.command("routing-screening-pipeline")
def routing_screening_pipeline(
    baseline_sheet_id: str,
    pi_noise_sheet_id: str,
    role_ce_sheet_id: str,
    pi_role_ce_sheet_id: str,
    max_samples: int = typer.Option(520, min=1, max=5000),
    compare_base: bool = typer.Option(False, "--compare-base"),
) -> None:
    """Run the fixed A/B/C/D regularizer screen and evaluate all cells on DEV."""
    from .pipeline import run_routing_screening_pipeline

    console.print_json(
        data=run_routing_screening_pipeline(
            sheet_ids={
                "A-baseline": baseline_sheet_id,
                "B-pi-noise": pi_noise_sheet_id,
                "C-role-ce": role_ce_sheet_id,
                "D-pi-role-ce": pi_role_ce_sheet_id,
            },
            max_samples=max_samples,
            compare_base=compare_base,
        )
    )


@app.command("resume-routing-screening-pipeline")
def resume_routing_screening_pipeline_command(
    pipeline_id: str | None = typer.Option(None, "--pipeline-id"),
) -> None:
    """Resume the latest interrupted A/B/C/D screen without repeating completed cells."""
    from .pipeline import resume_routing_screening_pipeline

    console.print_json(data=resume_routing_screening_pipeline(pipeline_id))


@app.command("geometry-screening-pipeline")
def geometry_screening_pipeline(
    baseline_run_id: str,
    noise_run_id: str,
    geometry_sheet_id: str,
    combined_sheet_id: str,
    source_pipeline_id: str,
    max_samples: int = typer.Option(520, min=1, max=5000),
) -> None:
    """Reuse A/B, train geometry E/F and compare all four on external DEV."""
    from .pipeline import run_geometry_screening_pipeline

    console.print_json(
        data=run_geometry_screening_pipeline(
            baseline_run_id=baseline_run_id,
            noise_run_id=noise_run_id,
            geometry_sheet_id=geometry_sheet_id,
            combined_sheet_id=combined_sheet_id,
            source_pipeline_id=source_pipeline_id,
            max_samples=max_samples,
        )
    )


@app.command("resume-geometry-screening-pipeline")
def resume_geometry_screening_pipeline_command(
    pipeline_id: str | None = typer.Option(None, "--pipeline-id"),
) -> None:
    """Resume E/F without retraining completed geometry cells."""
    from .pipeline import resume_geometry_screening_pipeline

    console.print_json(data=resume_geometry_screening_pipeline(pipeline_id))


@app.command("pair-pipeline")
def pair_pipeline(
    baseline_sheet_id: str,
    exotic_sheet_id: str,
    suite_id: str,
    max_samples: int = typer.Option(128, min=1, max=1000),
    compare_base: bool = typer.Option(False, "--compare-base"),
) -> None:
    """Train a fair pair, run one sealed suite, and persist compare-ready run IDs."""
    from .pipeline import run_pair_pipeline

    console.print_json(
        data=run_pair_pipeline(
            baseline_sheet_id=baseline_sheet_id,
            exotic_sheet_id=exotic_sheet_id,
            suite_id=suite_id,
            max_samples=max_samples,
            compare_base=compare_base,
        )
    )


@app.command("ablation-pipeline")
def ablation_pipeline(
    baseline_run_id: str,
    noise_run_id: str,
    geometry_sheet_id: str,
    combined_sheet_id: str,
    suite_id: str,
    max_samples: int = typer.Option(500, min=1, max=1000),
    compare_base: bool = typer.Option(False, "--compare-base"),
) -> None:
    """Run geometry-only and combined training, then a shared four-way sealed test."""
    from .pipeline import run_ablation_pipeline

    console.print_json(
        data=run_ablation_pipeline(
            baseline_run_id=baseline_run_id,
            noise_run_id=noise_run_id,
            geometry_sheet_id=geometry_sheet_id,
            combined_sheet_id=combined_sheet_id,
            suite_id=suite_id,
            max_samples=max_samples,
            compare_base=compare_base,
        )
    )


@app.command("literal-lock-pipeline")
def literal_lock_pipeline(
    baseline_sheet_id: str,
    pi_noise_sheet_id: str,
    pi_geometry_sheet_id: str,
    literal_lock_sheet_id: str,
    suite_id: str,
    dev_samples: int = typer.Option(520, min=1, max=5000),
    sealed_samples: int = typer.Option(520, min=1, max=1000),
    compare_base: bool = typer.Option(False, "--compare-base"),
) -> None:
    """Run the frozen A/B/F/G LiteralLock DEV and final sealed pipeline."""
    from .pipeline import run_literal_lock_pipeline

    console.print_json(
        data=run_literal_lock_pipeline(
            sheet_ids={
                "A-baseline": baseline_sheet_id,
                "B-pi-noise": pi_noise_sheet_id,
                "F-pi-geometry": pi_geometry_sheet_id,
                "G-pi-literal-lock": literal_lock_sheet_id,
            },
            suite_id=suite_id,
            dev_samples=dev_samples,
            sealed_samples=sealed_samples,
            compare_base=compare_base,
        )
    )


@app.command("resume-literal-lock-pipeline")
def resume_literal_lock_pipeline_command(
    pipeline_id: str | None = typer.Option(None, "--pipeline-id"),
) -> None:
    """Resume LiteralLock without retraining completed A/B/F/G cells."""
    from .pipeline import resume_literal_lock_pipeline

    console.print_json(data=resume_literal_lock_pipeline(pipeline_id))


@app.command("human-agentic-pipeline")
def human_agentic_pipeline(
    template_sheet_id: str,
    max_steps: int = typer.Option(400, min=100, max=10_000),
    dev_samples: int = typer.Option(520, min=1, max=5_000),
    sealed_samples: int = typer.Option(520, min=1, max=1_000),
    compare_base: bool = typer.Option(True, "--compare-base/--no-compare-base"),
) -> None:
    """Prepare and run a human-instruction 13-tool A/B/C/D/E experiment."""
    from .pipeline import prepare_human_agentic_sheets, run_human_agentic_pipeline
    from .workbook import WorkbookStore

    template = WorkbookStore().get(template_sheet_id).recipe
    prepared = prepare_human_agentic_sheets(template, max_steps=max_steps)
    console.print_json(
        data=run_human_agentic_pipeline(
            sheet_ids=prepared["sheet_ids"],
            suite_id=prepared["suite_id"],
            dev_samples=dev_samples,
            sealed_samples=sealed_samples,
            compare_base=compare_base,
        )
    )


@app.command("resume-human-agentic-pipeline")
def resume_human_agentic_pipeline_command(
    pipeline_id: str | None = typer.Option(None, "--pipeline-id"),
) -> None:
    """Resume the human-agentic pipeline without repeating completed cells."""
    from .pipeline import resume_human_agentic_pipeline

    console.print_json(data=resume_human_agentic_pipeline(pipeline_id))


@app.command("focused-human-pipeline")
def focused_human_pipeline(
    samples: int = typer.Option(520, min=1, max=1_000),
) -> None:
    """Run π/√2 full-noise, anchor geometry and π-LiteralLock on human tasks."""
    from .focused_pipeline import prepare_focused_sheets, run_focused_pipeline

    prepared = prepare_focused_sheets()
    console.print_json(data=run_focused_pipeline(prepared, samples=samples))


@app.command("resume-focused-human-pipeline")
def resume_focused_human_pipeline_command(
    pipeline_id: str | None = typer.Option(None, "--pipeline-id"),
) -> None:
    """Resume the latest focused human/strict/elastic experiment."""
    from .focused_pipeline import resume_focused_pipeline

    console.print_json(data=resume_focused_pipeline(pipeline_id))


@app.command("decimal-noise-control-pipeline")
def decimal_noise_control_pipeline(
    samples: int = typer.Option(520, min=1, max=1_000),
) -> None:
    """Compare π-clean30, sqrt(2)-clean30 and matched standard noise."""
    from .decimal_control_pipeline import (
        prepare_decimal_control_sheets,
        run_decimal_control_pipeline,
    )

    prepared = prepare_decimal_control_sheets()
    console.print_json(data=run_decimal_control_pipeline(prepared, samples=samples))


@app.command("resume-decimal-noise-control-pipeline")
def resume_decimal_noise_control_pipeline_command(
    pipeline_id: str | None = typer.Option(None, "--pipeline-id"),
) -> None:
    """Resume decimal/no-decimal controls without repeating completed cells."""
    from .decimal_control_pipeline import resume_decimal_control_pipeline

    console.print_json(data=resume_decimal_control_pipeline(pipeline_id))


@app.command("run-prepared-human-agentic-pipeline")
def run_prepared_human_agentic_pipeline(
    baseline_sheet_id: str,
    pi_noise_sheet_id: str,
    geometry_sheet_id: str,
    pi_geometry_sheet_id: str,
    human_intent_sheet_id: str,
    suite_id: str,
    dev_samples: int = typer.Option(520, min=1, max=5_000),
    sealed_samples: int = typer.Option(520, min=1, max=1_000),
    compare_base: bool = typer.Option(True, "--compare-base/--no-compare-base"),
) -> None:
    """Run an already audited human-agentic five-cell workbook."""
    from .pipeline import run_human_agentic_pipeline

    console.print_json(
        data=run_human_agentic_pipeline(
            sheet_ids={
                "A-baseline": baseline_sheet_id,
                "B-pi-noise": pi_noise_sheet_id,
                "C-geometry-only": geometry_sheet_id,
                "D-pi-geometry": pi_geometry_sheet_id,
                "E-human-intent": human_intent_sheet_id,
            },
            suite_id=suite_id,
            dev_samples=dev_samples,
            sealed_samples=sealed_samples,
            compare_base=compare_base,
        )
    )


@app.command()
def gui(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(17860),
    share: bool = typer.Option(False, help="Create a public Gradio share link."),
) -> None:
    """Launch the local browser GUI."""
    from .gui import launch_gui

    launch_gui(host=host, port=port, share=share)


@app.command("bfcl-setup")
def bfcl_setup(version: str = typer.Argument(..., help="v3 or v4")) -> None:
    """Install pinned official BFCL source plus compact deps in the project virtualenv."""
    from .bfcl import setup_bfcl_environment

    console.print_json(data=setup_bfcl_environment(version))


@app.command("bfcl-run")
def bfcl_run(config: Path) -> None:
    """Run a saved BFCL job through the local XPU or an OpenAI-compatible endpoint."""
    from .bfcl import run_bfcl

    console.print_json(data=run_bfcl(config))


@app.command("bfcl-score-only")
def bfcl_score_only(
    config: Path,
    category: list[str] | None = typer.Option(
        None,
        "--category",
        help="Optional BFCL v3 category to score; repeat the option for multiple categories.",
    ),
) -> None:
    """Resume official BFCL v3 scoring from already-complete prediction files."""
    from .bfcl import score_existing_bfcl

    console.print_json(data=score_existing_bfcl(config, categories=category))


@app.command("bfcl-v3-pair-pipeline")
def bfcl_v3_pair_pipeline(
    steps: int = typer.Option(400, min=100, max=2_000),
) -> None:
    """Train a fair rsLoRA/pi pair, then run official full BFCL v3 on three targets."""
    from .bfcl_pair_pipeline import run_pipeline

    console.print_json(data=run_pipeline(steps=steps))


@app.command("resume-bfcl-v3-pair-pipeline")
def resume_bfcl_v3_pair_pipeline(
    pipeline_id: str | None = typer.Argument(None),
    target_order: str = typer.Option(
        "pi-noise,baseline,base",
        help="Comma-separated evaluation order using pi-noise, baseline and base.",
    ),
) -> None:
    """Resume BFCL v3 evaluation without repeating completed fair-pair training."""
    from .bfcl_pair_pipeline import resume_pipeline

    console.print_json(data=resume_pipeline(pipeline_id, target_order=target_order))


@app.command("pi-export")
def pi_export(
    model_id: str,
    endpoint: str = typer.Option("http://127.0.0.1:8080/v1"),
    output: Path = typer.Option(DEFAULT_PI_OUTPUT),
) -> None:
    from .pi_export import write_pi_models

    destination = write_pi_models(model_id=model_id, endpoint=endpoint, output=output)
    console.print(f"[green]Wrote Pi provider config:[/green] {destination}")


@app.command()
def serve(
    model: str = typer.Option(
        "/mnt/git0/models/microLLM/LFM2.5-1.2B-Instruct", help="Base model path or registry id"
    ),
    adapter: str | None = typer.Option(None, help="Adapter directory or completed run directory"),
    device: str = typer.Option("auto", help="auto, xpu or cpu"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8080),
    handler: str = typer.Option(
        "liquid-lfm2",
        help="liquid-lfm2 or generic-openai-tools output parser",
    ),
    generation_timeout_seconds: float = typer.Option(
        180.0, min=1.0, help="Server-side maximum time for one generation."
    ),
    long_prompt_cache_threshold: int = typer.Option(
        2048,
        min=0,
        help="Disable KV cache above this prompt length to protect VRAM (0 disables it always).",
    ),
    xpu_memory_fraction: float = typer.Option(
        0.88,
        min=0.1,
        max=1.0,
        help="Hard fraction of XPU memory available to this model process.",
    ),
    default_max_new_tokens: int = typer.Option(
        1024, min=1, max=8192, help="Maximum generated tokens per request."
    ),
) -> None:
    """Serve base + adapter through an OpenAI-compatible endpoint for Pi."""
    from .server import serve as serve_gateway

    serve_gateway(
        model_path=Registry().resolve_model(model),
        adapter_path=adapter,
        device=device,
        host=host,
        port=port,
        handler=handler,
        generation_timeout_seconds=generation_timeout_seconds,
        long_prompt_cache_threshold=long_prompt_cache_threshold,
        xpu_memory_fraction=xpu_memory_fraction,
        default_max_new_tokens=default_max_new_tokens,
    )


if __name__ == "__main__":
    app()
