#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f "$PROJECT_ROOT/backend/.env" ]; then
    set -a
    . "$PROJECT_ROOT/backend/.env"
    set +a
fi

CLIMATE_DATA_ROOT="${CLIMATE_DATA_ROOT:-/tmp/climate-data}"
CLIMATE_MODEL_DIR="${CLIMATE_MODEL_DIR:-/tmp/climate-models}"
CLIMATE_METRICS_DIR="${CLIMATE_METRICS_DIR:-/tmp/climate-metrics}"
IMD_DATA_YEARS="${IMD_DATA_YEARS:-2025}"

export CLIMATE_DATA_ROOT
export CLIMATE_MODEL_DIR
export CLIMATE_METRICS_DIR
export IMD_RAINFALL_DIR="${IMD_RAINFALL_DIR:-${CLIMATE_DATA_ROOT}/rainfall}"
export IMD_TEMPERATURE_DIR="${IMD_TEMPERATURE_DIR:-${CLIMATE_DATA_ROOT}}"

set -f
set -- $IMD_DATA_YEARS
if [ "$#" -eq 0 ]; then
    echo "Startup error: IMD_DATA_YEARS must contain at least one year." >&2
    exit 1
fi
for year in "$@"; do
    case "$year" in
        [0-9][0-9][0-9][0-9]) ;;
        *)
            echo "Startup error: invalid year in IMD_DATA_YEARS: $year" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$CLIMATE_DATA_ROOT" "$CLIMATE_MODEL_DIR" "$CLIMATE_METRICS_DIR"
echo "Checking official IMD grids for ${IMD_DATA_YEARS} under $CLIMATE_DATA_ROOT"
if ! python scripts/download_imd_grids.py --years "$@" --data-root "$CLIMATE_DATA_ROOT"; then
    echo "Startup error: official IMD grids could not be downloaded or verified. See the downloader error above." >&2
    exit 1
fi

if ! python -m scripts.climate_model_bundle verify; then
    if [ -z "${CLIMATE_MODEL_BUNDLE_URL:-}" ]; then
        echo "Startup error: forecast model artifacts are missing or invalid, and CLIMATE_MODEL_BUNDLE_URL is not set." >&2
        echo "Train the models locally, package them with python -m scripts.climate_model_bundle, and provide the public bundle URL." >&2
        exit 1
    fi
    echo "Installing the prepared forecast model bundle."
    if ! python -m scripts.climate_model_bundle install --url "$CLIMATE_MODEL_BUNDLE_URL"; then
        echo "Startup error: the forecast model bundle could not be downloaded or validated." >&2
        exit 1
    fi
fi

if ! python -m scripts.climate_model_bundle verify; then
    echo "Startup error: forecast model verification failed after preparation." >&2
    exit 1
fi

cd "$PROJECT_ROOT/backend"
exec python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-10000}"
