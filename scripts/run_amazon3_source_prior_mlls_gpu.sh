#!/bin/bash
# Resumable six-cell BCTS+MLLS follow-up. Existing v1 results are read-only.

set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="outputs/student/amazon3_source_prior_transfer_mlls_v1"
LOG_FILE="${OUT_DIR}/run_log.txt"
INVALID_DIR="${OUT_DIR}/_invalid"
METHOD="estimated_prior_mlls_bcts"
LABEL_BUDGETS=(10 50)
SEEDS=(42 123 456)
EXPECTED=6

PYTHON_BIN="python"
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

mkdir -p "$OUT_DIR" "$INVALID_DIR"

log() {
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[$ts] $1" | tee -a "$LOG_FILE"
}

result_path() {
  echo "${OUT_DIR}/amazon3_source-prior-transfer_${METHOD}_L${1}_s${2}.json"
}

cell_is_valid() {
  "$PYTHON_BIN" scripts/analyse_amazon3_source_prior_mlls.py \
    --check-cell "$1" --labels-per-class "$2" --seed "$3" >/dev/null 2>&1
}

quarantine_invalid() {
  local path="$1"
  if [ ! -e "$path" ]; then
    return
  fi
  local dest
  dest="${INVALID_DIR}/$(date -u +"%Y%m%dT%H%M%SZ")_$(basename "$path")"
  mv "$path" "$dest"
  log "QUARANTINE invalid -> $dest"
}

log "=== BCTS+MLLS preflight begin ==="
if ! "$PYTHON_BIN" tests/test_amazon3_source_prior.py 2>&1 | tee -a "$LOG_FILE"; then
  log "Preflight failed: focused tests"
  exit 1
fi

for label_budget in "${LABEL_BUDGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    if ! out=$("$PYTHON_BIN" scripts/train_amazon3_source_prior.py \
      --method "$METHOD" --labels-per-class "$label_budget" --seed "$seed" --dry-run 2>&1); then
      log "Preflight failed: dry-run L=$label_budget seed=$seed"
      echo "$out" | tail -20 | tee -a "$LOG_FILE"
      exit 1
    fi
    echo "$out" | grep "BCTS+MLLS estimate" | tee -a "$LOG_FILE"
  done
done
log "=== BCTS+MLLS preflight OK ==="

started=0
skipped=0
completed=0
failures=0

for label_budget in "${LABEL_BUDGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    path="$(result_path "$label_budget" "$seed")"
    if [ -e "$path" ] && cell_is_valid "$path" "$label_budget" "$seed"; then
      log "SKIP valid L=$label_budget seed=$seed"
      skipped=$((skipped + 1))
      continue
    fi
    if [ -e "$path" ]; then
      quarantine_invalid "$path"
    fi

    log "START L=$label_budget seed=$seed"
    started=$((started + 1))
    if ! "$PYTHON_BIN" scripts/train_amazon3_source_prior.py \
      --method "$METHOD" \
      --labels-per-class "$label_budget" \
      --seed "$seed" \
      --output-dir "$OUT_DIR" 2>&1 | tee -a "$LOG_FILE"; then
      log "FAILED training L=$label_budget seed=$seed"
      failures=$((failures + 1))
      continue
    fi
    if ! cell_is_valid "$path" "$label_budget" "$seed"; then
      log "FAILED validation L=$label_budget seed=$seed"
      quarantine_invalid "$path"
      failures=$((failures + 1))
      continue
    fi
    completed=$((completed + 1))
    log "OK L=$label_budget seed=$seed"
  done
done

valid=$((skipped + completed))
log "=== runner end started=$started skipped=$skipped completed=$completed failures=$failures ==="
if [ "$failures" -ne 0 ] || [ "$valid" -ne "$EXPECTED" ]; then
  echo "BCTS+MLLS grid incomplete; rerun to resume." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/analyse_amazon3_source_prior_mlls.py | tee -a "$LOG_FILE"
log "All six BCTS+MLLS cells complete and analysed."
