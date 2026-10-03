#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python3"
RUN_ROOT="${REPO_ROOT}/runs/trinity_v13_strict_genetic_prod"
MODE="${1:-prepare}"
shift || true

EXECUTE_XPU=0
while (($#)); do
    case "$1" in
        --execute-xpu)
            EXECUTE_XPU=1
            shift
            ;;
        --run-root)
            if (($# < 2)); then
                echo "--run-root richiede un percorso" >&2
                exit 2
            fi
            RUN_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Argomento sconosciuto: $1" >&2
            exit 2
            ;;
    esac
done

RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/pipeline_$(date -u +%Y%m%d-%H%M%S)_${MODE}.log"

export PYTHONUNBUFFERED=1
export ZE_AFFINITY_MASK=0
export ONEAPI_DEVICE_SELECTOR=level_zero:0

require_xpu_ack() {
    if ((EXECUTE_XPU != 1)); then
        echo "Fase XPU bloccata. Rieseguire con --execute-xpu dopo il preflight." >&2
        exit 2
    fi
}

run_logged() {
    echo "+ $*" | tee -a "${LOG_FILE}"
    "$@" 2>&1 | tee -a "${LOG_FILE}"
}

prepare() {
    run_logged "${PYTHON}" scripts/prepare_trinity_v13_campaign.py --run-root "${RUN_ROOT}"
}

tournament() {
    require_xpu_ack
    run_logged "${PYTHON}" scripts/run_trinity_v13_strict_genetic.py \
        --output-dir "${RUN_ROOT}" \
        --population 5 \
        --round-steps 20,35 \
        --micro-rows 420 \
        --tournament-seed 20260823 \
        --final-steps 240 \
        --execute-xpu
    run_logged "${PYTHON}" scripts/prepare_trinity_v13_campaign.py \
        --run-root "${RUN_ROOT}" \
        --materialize-adapter
}

benchmarks() {
    require_xpu_ack
    run_logged "${PYTHON}" scripts/run_trinity_v13_benchmarks.py \
        --run-root "${RUN_ROOT}" \
        --suite all \
        --xpu-memory-fraction 0.70 \
        --execute-xpu
}

report() {
    run_logged "${PYTHON}" scripts/generate_trinity_v13_reports.py --run-root "${RUN_ROOT}"
}

cd "${REPO_ROOT}"
case "${MODE}" in
    prepare|plan)
        prepare
        report
        ;;
    tournament)
        prepare
        tournament
        ;;
    benchmarks)
        prepare
        benchmarks
        report
        ;;
    report)
        prepare
        report
        ;;
    all)
        require_xpu_ack
        prepare
        tournament
        benchmarks
        report
        ;;
    *)
        echo "Uso: $0 {prepare|plan|tournament|benchmarks|report|all} [--run-root PATH] [--execute-xpu]" >&2
        exit 2
        ;;
esac

echo "Completato: mode=${MODE} run_root=${RUN_ROOT}" | tee -a "${LOG_FILE}"
