from __future__ import annotations

import csv
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import runs_dir

CAMPAIGN_NAME = "trinity_v13_strict_genetic_prod"
CAMPAIGN_SCHEMA = "trinity-v13-campaign-v1"
REQUIRED_DIRECTORIES = (
    "docs",
    "roadmap",
    "data",
    "genetic_tournament",
    "final_training",
    "adapter-final",
    "benchmarks/sealed/raw",
    "benchmarks/bfcl_v3/raw",
    "benchmarks/bfcl_v4/raw",
    "graphs",
    "logs",
)


def default_campaign_root() -> Path:
    return runs_dir() / CAMPAIGN_NAME


def _write_if_missing(path: Path, text: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _campaign_readme() -> str:
    return """# TRINITY V13 strict genetic production run

Questa cartella e la radice autosufficiente dell'esperimento V13. Contiene protocollo,
roadmap, dati derivati, torneo genetico, training finale, benchmark e grafici. Nessun
risultato viene inserito manualmente nei report: CSV, Markdown e PNG sono rigenerati dai
file grezzi conservati in `benchmarks/*/raw/` e dallo stato del torneo.

## Struttura

- `docs/`: disegno sperimentale, glossario delle metriche e policy anti-contaminazione.
- `roadmap/`: percorso decisionale e procedura di riproducibilita.
- `data/`: manifest dei corpus ex-novo; non contiene SEALED o BFCL.
- `genetic_tournament/`: geni, micro-training, doppio seed e ranking per round.
- `final_training/`: training completo del gene vincitore e selezione finale fra seed.
- `adapter-final/`: copia immutabile dell'adapter selezionato, creata solo a training finito.
- `benchmarks/`: risultati finali SEALED, BFCL v3 e BFCL v4, separati per protocollo.
- `graphs/`: PNG generati esclusivamente dai risultati presenti in `benchmarks/`.
- `logs/`: log dei launcher e degli evaluator.

Il launcher non usa XPU per default. Ogni fase che carica il modello richiede una conferma
esplicita e non effettua retry automatici dopo un errore Level Zero o device loss.
"""


def _experiment_design() -> str:
    return r"""# Disegno sperimentale V13

## Obiettivo

Selezionare ex-novo un tuning LoRA stabile per LFM2.5-1.2B-Instruct, con priorita a strict
AST, routing no-tool, multi-step, web multi-hop e memory, senza sacrificare formato e
rehearsal generale, integrando sinergicamente tutte le ottimizzazioni esotiche.

## Regola di selezione

Il torneo usa soltanto DEV locali sintetici preregistrati e disgiunti dal training. SEALED,
BFCL v3 e BFCL v4 ufficiali non vengono interrogati durante la selezione: ripetere micro-ritagli
dei test ufficiali trasformerebbe il test in validation e renderebbe il risultato ottimistico.

Ogni gene viene valutato con fitness robusta su seed 6492 e 8778.

## Rotazione DEV

- round 0: DEV-A, 10 casi per ciascuna delle 6 categorie;
- round 1: 50% DEV-A + 50% DEV-B per categoria.

I mix sono deterministici e salvati accanto a ciascun candidato. I nuovi record impediscono
la fossilizzazione sul fold precedente; la meta conservata permette di misurare continuita.

## Architettura Multi-Regularizer Unificata

Ogni candidato attiva contemporaneamente tutte le tecniche esotiche con parametri continui
accordati congiuntamente:
- 3D Boolean Polytope margin, pesi arity/action/syntax, refusal loss;
- $\pi$-Noise multi-granulare (digit pairs / diff / single digit) con inviluppo coseno;
- Geometria Riemanniana/Relazionale token-level su layer profondo (-1);
- Role Loss pesata (delimiter, tool name, argument key/value) con TEMH turn-horizon beta;
- State Binding Anchor per coerenza memoria a lungo raggio.

## Sicurezza memoria

- frazione XPU massima per processo: 0.70;
- sequenza massima: 1408 token;
- LoRA rank: 16 (alpha: 32, rsLoRA);
- target LoRA: attention;
- micro-batch 1 e gradient checkpointing;
- nessun retry automatico dopo device loss.
"""


def _roadmap() -> str:
    return """# Roadmap e riproducibilita V13

## Come si e arrivati a V13

1. Le stime storiche precedenti a V10 non sono usate come verita di selezione.
2. Anche per V10-V12 conta la provenienza dell'evaluator: alcuni JSON furono prodotti da
   stimatori locali Python che sovrastimavano alcune categorie.
3. I vecchi corpus sintetici V11/V12 avevano duplicazione estrema; V13 genera esempi unici,
   coppie negative e tre DEV senza overlap.
4. Un tentativo V13 precedente ha causato device loss durante preference training. Quella
   run resta separata e fallita; non viene ripresa e non fornisce un vincitore.
5. Il nuovo protocollo limita memoria e regolarizzatori e richiede conferma esplicita XPU.

## Esecuzione riproducibile

1. `scripts/run_trinity_v13_pipeline.sh prepare`
2. revisione di `docs/`, manifest del corpus e preflight memoria/disco;
3. `scripts/run_trinity_v13_pipeline.sh tournament --execute-xpu`
4. materializzazione di `adapter-final/` solo dopo il completamento dei due seed finali;
5. `scripts/run_trinity_v13_pipeline.sh benchmarks --execute-xpu`
6. `scripts/run_trinity_v13_pipeline.sh report`

Ogni comando scrive sotto questa run. I report finali devono includere hash/config, seed,
versione metrica, commit BFCL e indicazione `publishable` o `diagnostic`.

## Criteri di arresto

Il launcher si arresta al primo errore. Device loss, memoria insufficiente, adapter mancante,
provenienza incompleta o benchmark parziale non vengono convertiti in successo. Un BFCL v4
eseguito sul subset ufficiale congelato al 25% e diagnostico e non equivale al leaderboard
full; deve essere etichettato come tale nel report.
"""


def _metrics_and_provenance() -> str:
    return """# Metriche e provenienza

## Regola storica

Alcune vecchie stime Python erano troppo alte in certe categorie. Il fatto che un JSON
contenga un punteggio non lo rende affidabile, e il solo numero di versione (anche V10+)
non basta: servono evaluator, commit, config, numero di casi e output grezzo.

## BFCL

- `strict` e il valore principale per chiamate, nomi, chiavi, tipi e valori esatti;
- `elastic` e diagnostico e non puo sostituire strict;
- il gap elastic-strict viene sempre mostrato e penalizzato nella fitness locale;
- la micro-media pesata sui casi non va chiamata score leaderboard se l'aggregazione
  ufficiale usa macro-famiglie differenti;
- un subset eseguito dall'evaluator ufficiale e comunque `partial/diagnostic`.

Sono accettati nel report V13 soltanto score prodotti dal runner BFCL pinned e accompagnati
da config/state. I risultati locali legacy non vengono importati automaticamente.

## SEALED

SEALED-520 e una suite locale frozen, non un leaderboard pubblico. Il report conserva
routing, formato, strict ed elastic per stratum. SEALED e usato una sola volta dopo il
freeze dell'adapter e mai per scegliere geni o seed.

## Protezione dalle regressioni

La fitness include rehearsal e il minimo fra routing, formato e rehearsal. In questo modo
il negative training non puo guadagnare soltanto su strict facendo crollare una delle tre
capacita di controllo. Il seed peggiore e la deviazione fra i due seed partecipano al
ranking robusto.
"""


def _tournament_plan() -> str:
    return r"""# Piano dettagliato del torneo genetico V13

## Ampiezza e livelli

| Livello | Candidati previsti | SFT step | DEV | Micro-training |
|---|---:|---:|---|---:|
| 0 | 5 candidati iniziali | 20 | A | 5 x 2 seed |
| 1 | top 3 + 1 crossover/mutazione | 35 | 50% A + 50% B | 4 x 2 seed |

Ogni cella usa 420 record, micro-batch 1, gradient checkpointing e i seed 6492/8778.
Nessun gene disabilita i moduli esotici: la popolazione esplora e accorda congiuntamente
3D-Boolean, $\pi$-noise, geometria Riemanniana, role loss (TEMH) e state-binding.

## Spazio genetico

- LoRA rank: 16, alpha: 32, rsLoRA, sequence length fissa 1408, target attention;
- learning rate log-uniforme 7.0e-5 - 2.2e-4, weight decay 0.005-0.05, warmup 0.03-0.08;
- $\pi$-noise: sorgente $\pi$, digit pairs / diff / single digit, alpha 1.8-2.6, modulazione 0.10-0.20, envelope coseno, clean tail;
- Geometria Riemanniana/Relazionale: relational mode, layer -1, peso 0.007-0.015, margin 0.32-0.45, 32 sample tokens;
- Boolean 3D: peso 0.006-0.018, polytope margin 0.28-0.45, pesi dimensionali action/arity/syntax, 48 sample tokens, layer -1;
- Role loss & TEMH: delimiter, tool name, argument key, argument value e turn-horizon beta (0.40-0.70);
- State binding: anchor consistency per context multi-turn;
- Data curriculum: category weights, ordinamento (interleaved/hard-first/rehearsal-spaced) e repeat focus.

## Fitness e doppio seed

Fitness per seed:

`0.35 strict + 0.15 routing + 0.10 MCC_scaled + 0.20 agentic/multistep + 0.10 format +
0.10 rehearsal + 0.05 min(routing,format,rehearsal) - 0.10 elastic_strict_gap`

Aggregazione dei due seed:

`0.70 media + 0.30 seed_peggiore - 0.15 deviazione_standard`

## Training completo e benchmark ufficiali

Il gene vincitore viene addestrato su record rehearsal unici + record targeted,
per due seed e 240 SFT step. Il seed finale viene scelto sul composito local DEV A/B/C.
Solo dopo il freeze e la copia in `adapter-final/` si eseguono:

1. SEALED-520 frozen con sottodettagli per stratum;
2. BFCL v3 full con evaluator ufficiale Berkeley pinned;
3. BFCL v4 con evaluator ufficiale Berkeley pinned sul subset frozen 25%, marcato partial/diagnostic.
"""


def _benchmark_readme() -> str:
    return """# Benchmark finali V13

Questa cartella viene popolata soltanto dopo la selezione e il training completo del
vincitore. I benchmark non partecipano alla fitness genetica.

- `sealed/`: suite locale frozen SEALED-520, con metriche complessive e per stratum.
- `bfcl_v3/`: evaluator BFCL v3 ufficiale pinned, protocollo full se completato.
- `bfcl_v4/`: evaluator BFCL v4 ufficiale; il subset frozen 25% resta diagnostico/partial.

Ogni sottocartella contiene `raw/` per l'output originale, `category_scores.csv` per la
normalizzazione e `REPORT.md` per provenienza e riepilogo. I PNG corrispondenti sono in
`../graphs/`.
"""


def _graphs_readme() -> str:
    return """# Grafici V13

I PNG sono generati da `scripts/generate_trinity_v13_reports.py`. Lo script non contiene
punteggi hardcoded: se un benchmark manca, non inventa il grafico e lo marca pending nel
report complessivo.
"""


def create_campaign_layout(root: Path | None = None) -> dict[str, Any]:
    root = (root or default_campaign_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for relative in REQUIRED_DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)

    _write_if_missing(root / "README.md", _campaign_readme())
    _write_if_missing(root / "docs" / "EXPERIMENT_DESIGN.md", _experiment_design())
    _write_if_missing(root / "docs" / "METRICS_AND_PROVENANCE.md", _metrics_and_provenance())
    _write_if_missing(root / "docs" / "GENETIC_TOURNAMENT_PLAN.md", _tournament_plan())
    _write_if_missing(root / "roadmap" / "ROADMAP_AND_REPRODUCIBILITY.md", _roadmap())
    _write_if_missing(root / "benchmarks" / "README.md", _benchmark_readme())
    _write_if_missing(root / "graphs" / "README.md", _graphs_readme())
    for suite in ("sealed", "bfcl_v3", "bfcl_v4"):
        _write_if_missing(
            root / "benchmarks" / suite / "README.md",
            f"# {suite}\n\nOutput grezzi in `raw/`; CSV e report vengono generati dopo l'evaluation.\n",
        )

    manifest_path = root / "campaign_manifest.json"
    manifest = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign": CAMPAIGN_NAME,
        "root": str(root),
        "created_at": datetime.now(UTC).isoformat(),
        "required_directories": list(REQUIRED_DIRECTORIES),
        "selection_benchmark_policy": "local-dev-only",
        "final_benchmarks": ["sealed-520-frozen", "bfcl-v3-pinned", "bfcl-v4-pinned"],
        "xpu_execution_default": "disabled",
    }
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest


def validate_campaign_layout(root: Path) -> dict[str, Any]:
    root = root.resolve()
    missing = [relative for relative in REQUIRED_DIRECTORIES if not (root / relative).is_dir()]
    return {
        "status": "ready" if not missing else "incomplete",
        "root": str(root),
        "missing_directories": missing,
    }


def materialize_selected_adapter(root: Path) -> dict[str, Any]:
    """Copy the selected full-training adapter into the conventional root location.

    Existing adapter artifacts are never overwritten. This keeps repeated orchestration
    commands from silently changing the identity of a reported model.
    """

    root = root.resolve()
    selection_path = root / "final_training" / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Final selection is missing: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source = Path(str(selection["selected_adapter"])).resolve()
    if not (source / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Selected adapter is incomplete: {source}")
    destination = root / "adapter-final"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    identity = {
        "selected_seed": selection["selected_seed"],
        "source": str(source),
        "destination": str(destination),
        "selection_policy": selection.get("selection_policy"),
        "materialized_at": datetime.now(UTC).isoformat(),
    }
    (root / "final_training" / "adapter_identity.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    return identity


def _first_json_object(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _bfcl_family(category: str, version: str) -> str:
    value = category.lower()
    if "memory" in value or "web_search" in value:
        return "agentic"
    if "multi_turn" in value:
        return "multi_turn"
    if "live" in value:
        return "live"
    if version == "v3" and "exec" in value:
        return "executable"
    if "format_sensitivity" in value:
        return "format_sensitivity"
    return "non_live"


def read_bfcl_category_scores(score_root: Path, version: str) -> list[dict[str, Any]]:
    if version not in {"v3", "v4"}:
        raise ValueError(f"Unsupported BFCL version: {version}")
    prefix = f"BFCL_{version}_"
    rows: list[dict[str, Any]] = []
    for path in sorted(score_root.rglob(f"{prefix}*_score.json")):
        payload = _first_json_object(path)
        if not payload or "total_count" not in payload or "correct_count" not in payload:
            continue
        total = int(payload["total_count"])
        correct = int(payload["correct_count"])
        category = path.name.removeprefix(prefix).removesuffix("_score.json")
        rows.append(
            {
                "category": category,
                "family": _bfcl_family(category, version),
                "total": total,
                "correct": correct,
                "accuracy": correct / max(1, total),
                "metric": "official_accuracy",
                "source": str(path.resolve()),
            }
        )
    return rows


def read_sealed_category_scores(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    predictions = list(metrics.get("agent_predictions") or [])
    if not predictions:
        total = int(metrics.get("agent_eval_samples") or metrics.get("total_cases") or 0)
        if total == 0:
            return []
        return [
            {
                "category": "overall-only-legacy",
                "total": total,
                "strict_success": None,
                "elastic_success": None,
                "routing_accuracy": float(
                    metrics.get("agent_tool_decision_accuracy")
                    or metrics.get("balanced_overall_accuracy", 0.0) / 100.0
                ),
                "format_accuracy": float(
                    metrics.get("agent_expected_format_accuracy")
                    or metrics.get("format_compliance_accuracy", 0.0) / 100.0
                ),
                "strict_accuracy": None,
                "elastic_accuracy": None,
                "source": str(path.resolve()),
            }
        ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        metadata = item.get("evaluation_metadata") or {}
        buckets[str(metadata.get("stratum") or metadata.get("category") or "unclassified")].append(
            item
        )
    rows = []
    for category, selected in sorted(buckets.items()):
        total = len(selected)
        strict = sum(bool(item.get("strict_human_task_success")) for item in selected)
        elastic = sum(bool(item.get("elastic_human_task_success")) for item in selected)
        routing = sum(bool(item.get("tool_decision_correct")) for item in selected)
        formatted = sum(bool(item.get("expected_format_correct")) for item in selected)
        rows.append(
            {
                "category": category,
                "total": total,
                "strict_success": strict,
                "elastic_success": elastic,
                "routing_success": routing,
                "format_success": formatted,
                "strict_accuracy": strict / total,
                "elastic_accuracy": elastic / total,
                "routing_accuracy": routing / total,
                "format_accuracy": formatted / total,
                "source": str(path.resolve()),
            }
        )
    return rows


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _percent(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _render_bar_chart(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    series: list[tuple[str, str]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [str(row["category"]).replace("_", " ") for row in rows]
    width = 0.78 / max(1, len(series))
    x = np.arange(len(rows))
    figure_width = max(12.0, 0.62 * len(rows) + 5.0)
    fig, ax = plt.subplots(figsize=(figure_width, 8.5), dpi=180)
    palette = ("#1D4ED8", "#059669", "#D97706", "#7C3AED")
    for index, (field, label) in enumerate(series):
        values = [math.nan if row.get(field) is None else 100.0 * float(row[field]) for row in rows]
        offset = (index - (len(series) - 1) / 2.0) * width
        bars = ax.bar(x + offset, values, width, label=label, color=palette[index])
        for bar, value in zip(bars, values):
            if not math.isnan(value):
                ax.annotate(
                    f"{value:.1f}",
                    (bar.get_x() + bar.get_width() / 2.0, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=max(1, len(series)))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_tournament_chart(path: Path, state: dict[str, Any]) -> bool:
    rows = []
    for round_item in state.get("rounds") or []:
        for gene in round_item.get("genes") or []:
            rows.append(
                {
                    "round": int(round_item["round"]),
                    "gene": str(gene["gene"]["id"]),
                    "fitness": float(gene["robust_fitness"]),
                }
            )
    if not rows:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    genes = sorted({item["gene"] for item in rows})
    for gene in genes:
        selected = sorted((item for item in rows if item["gene"] == gene), key=lambda x: x["round"])
        ax.plot(
            [item["round"] for item in selected],
            [100.0 * item["fitness"] for item in selected],
            marker="o",
            linewidth=2,
            label=gene,
        )
    ax.set_title("TRINITY V13 - fitness robusta per round (doppio seed)", fontweight="bold")
    ax.set_xlabel("Round genetico")
    ax.set_ylabel("Robust fitness (%)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _latest_file(root: Path, names: Iterable[str]) -> Path | None:
    candidates: list[Path] = []
    for name in names:
        candidates.extend(root.rglob(name))
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _field, label in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, separator]
    for row in rows:
        values = []
        for field, _label in columns:
            value = row.get(field)
            values.append(_percent(value) if field.endswith("accuracy") else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_campaign_reports(root: Path | None = None) -> dict[str, Any]:
    root = (root or default_campaign_root()).resolve()
    create_campaign_layout(root)
    status: dict[str, Any] = {"root": str(root), "benchmarks": {}, "graphs": []}

    sealed_source = _latest_file(
        root / "benchmarks" / "sealed" / "raw",
        ("sealed_520_detailed.json", "metrics.json", "sealed_520_results.json"),
    )
    if sealed_source:
        rows = read_sealed_category_scores(sealed_source)
        write_rows_csv(root / "benchmarks" / "sealed" / "category_scores.csv", rows)
        report = (
            "# SEALED-520 frozen\n\n"
            "Provenienza: suite locale frozen OOD; non usata nel torneo genetico.\n\n"
            + _markdown_table(
                rows,
                [
                    ("category", "Categoria"),
                    ("total", "N"),
                    ("strict_accuracy", "Strict"),
                    ("elastic_accuracy", "Elastic"),
                    ("routing_accuracy", "Routing"),
                    ("format_accuracy", "Formato"),
                ],
            )
            + "\n"
        )
        (root / "benchmarks" / "sealed" / "REPORT.md").write_text(report, encoding="utf-8")
        chart = root / "graphs" / "sealed_520_by_category.png"
        _render_bar_chart(
            chart,
            rows,
            title="TRINITY V13 - SEALED-520 per categoria",
            series=[
                ("strict_accuracy", "Strict"),
                ("elastic_accuracy", "Elastic"),
                ("routing_accuracy", "Routing"),
                ("format_accuracy", "Formato"),
            ],
        )
        status["graphs"].append(str(chart))
        status["benchmarks"]["sealed"] = {"status": "ready", "rows": len(rows)}
    else:
        status["benchmarks"]["sealed"] = {"status": "pending"}

    for version in ("v3", "v4"):
        suite = f"bfcl_{version}"
        raw = root / "benchmarks" / suite / "raw"
        rows = read_bfcl_category_scores(raw, version)
        if rows:
            write_rows_csv(root / "benchmarks" / suite / "category_scores.csv", rows)
            total = sum(int(row["total"]) for row in rows)
            correct = sum(int(row["correct"]) for row in rows)
            provenance = (
                "evaluator BFCL pinned; la pubblicabilita dipende da config/state e dal completamento "
                "della suite. Un subset resta diagnostico."
            )
            report = (
                f"# BFCL {version}\n\nProvenienza: {provenance}\n\n"
                f"Micro-aggregato categorie presenti: **{correct}/{total} ({100 * correct / max(1, total):.2f}%)**. "
                "Non sostituisce l'aggregazione macro ufficiale del leaderboard.\n\n"
                + _markdown_table(
                    rows,
                    [
                        ("category", "Categoria"),
                        ("family", "Famiglia"),
                        ("total", "N"),
                        ("correct", "Corretti"),
                        ("accuracy", "Accuracy ufficiale"),
                    ],
                )
                + "\n"
            )
            (root / "benchmarks" / suite / "REPORT.md").write_text(report, encoding="utf-8")
            chart = root / "graphs" / f"bfcl_{version}_by_category.png"
            _render_bar_chart(
                chart,
                rows,
                title=f"TRINITY V13 - BFCL {version} per categoria",
                series=[("accuracy", "Accuracy ufficiale")],
            )
            status["graphs"].append(str(chart))
            status["benchmarks"][suite] = {
                "status": "ready",
                "rows": len(rows),
                "micro_accuracy": correct / max(1, total),
            }
        else:
            status["benchmarks"][suite] = {"status": "pending"}

    tournament_state = root / "genetic_tournament" / "state.json"
    if tournament_state.is_file():
        state = json.loads(tournament_state.read_text(encoding="utf-8"))
        chart = root / "graphs" / "genetic_tournament_fitness.png"
        if _render_tournament_chart(chart, state):
            status["graphs"].append(str(chart))

    lines = [
        "# Report complessivo TRINITY V13",
        "",
        "Il torneo usa esclusivamente DEV locali. SEALED e BFCL sono valutazioni finali.",
        "",
        "| Benchmark | Stato | Dettaglio |",
        "|---|---|---|",
    ]
    for suite, item in status["benchmarks"].items():
        detail = f"{item.get('rows', 0)} categorie" if item["status"] == "ready" else "non eseguito"
        lines.append(f"| {suite} | {item['status']} | {detail} |")
    lines.extend(
        [
            "",
            (
                "I punteggi storici non vengono copiati in questo report. I file pending "
                "restano tali finche non sono presenti output grezzi V13."
            ),
        ]
    )
    (root / "benchmarks" / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "benchmarks" / "report_manifest.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return status
