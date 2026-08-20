#!/bin/bash
# Resumable Amazon-3 source-prior-transfer GPU grid.
#
# Student-level replication under a source-induced non-uniform prior (see
# scripts/train_amazon3_source_prior.py for the full experimental design).
# 4 methods (scale_only, uniform_prior, estimated_prior_bbse, oracle_prior)
# x 2 label budgets (10, 50) x 3 seeds (42, 123, 456) = 24 GPU student runs.
# No teacher queries: guided-teacher grouped logits are derived from the
# existing cached 5-class Amazon logits.
#
# Preflight (before any GPU work):
#   1. Run the new experiment tests plus other relevant existing tests.
#   2. Run all 24 cells in --dry-run mode and print a preflight table
#      (budget, seed, estimated prior, L1 error, condition number, solver
#      residual, split counts, reference/working/test identity hashes).
#   3. Statically confirm the new experiment modules contain no teacher-query
#      code paths.
#   4. Confirm every result path is new (a fresh output directory).
#
# If the preflight passes (poor estimation is still a valid scientific
# result; only implementation failures stop the run), the runner executes
# the 24-cell grid, validating each cell before treating it as complete and
# quarantining invalid/partial files instead of silently overwriting them.

set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="outputs/student/amazon3_source_prior_transfer_v1"
LOG_FILE="${OUT_DIR}/run_log.txt"
INVALID_DIR="${OUT_DIR}/_invalid"
METHODS=(scale_only uniform_prior estimated_prior_bbse oracle_prior)
LABEL_BUDGETS=(10 50)
SEEDS=(42 123 456)

run_failures=0
run_started=0
run_skipped=0
run_completed=0

mkdir -p "$OUT_DIR" "$INVALID_DIR"

log() {
  local msg="$1"
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[$ts] $msg" | tee -a "$LOG_FILE"
}

result_path() {
  local method="$1"
  local label_budget="$2"
  local seed="$3"
  echo "${OUT_DIR}/amazon3_source-prior-transfer_${method}_L${label_budget}_s${seed}.json"
}

cell_is_valid() {
  local path="$1"
  local method="$2"
  local label_budget="$3"
  local seed="$4"
  uv run python scripts/analyse_amazon3_source_prior.py \
    --check-cell "$path" \
    --method "$method" \
    --labels-per-class "$label_budget" \
    --seed "$seed" >/dev/null 2>&1
}

quarantine_invalid() {
  local path="$1"
  if [ ! -e "$path" ]; then
    return
  fi
  local base stamp dest
  base="$(basename "$path")"
  stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  dest="${INVALID_DIR}/${stamp}_${base}"
  mv "$path" "$dest"
  log "QUARANTINE invalid -> $dest"
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
preflight() {
  log "=== preflight begin ==="

  log "Preflight 1/4: running relevant tests"
  if ! uv run pytest tests/test_amazon3_source_prior.py tests/test_prior_alignment.py \
      tests/test_student_prior_misspecification.py tests/test_student.py \
      tests/test_dataset_identity.py -q 2>&1 | tee -a "$LOG_FILE"; then
    log "=== preflight FAILED: tests did not pass ==="
    return 1
  fi

  log "Preflight 2/4: no-teacher-query static check"
  if grep -RInE "teachers\.llm|teachers\.extraction\.complete|openai\.OpenAI\(|vllm" \
      scripts/train_amazon3_source_prior.py src/student/prior_estimation.py; then
    log "=== preflight FAILED: forbidden teacher-query reference found ==="
    return 1
  fi
  log "  no teacher-query code paths found in the new experiment modules"

  log "Preflight 3/4: confirming output directory is new/isolated"
  local pre_existing
  pre_existing=$(find "$OUT_DIR" -maxdepth 1 -name "amazon3_source-prior-transfer_*.json" | wc -l)
  log "  pre-existing result files under $OUT_DIR: $pre_existing"

  log "Preflight 4/4: running all 24 cells in --dry-run mode"
  printf "%-22s %-6s %-6s %-32s %-10s %-8s %-10s\n" \
    "method" "L" "seed" "estimated_prior" "l1_error" "cond" "residual" | tee -a "$LOG_FILE"
  local preflight_failures=0
  for label_budget in "${LABEL_BUDGETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for method in "${METHODS[@]}"; do
        local out
        if ! out=$(uv run python scripts/train_amazon3_source_prior.py \
            --method "$method" --labels-per-class "$label_budget" --seed "$seed" --dry-run 2>&1); then
          log "DRY-RUN FAILED method=$method L=$label_budget seed=$seed"
          echo "$out" | tail -20 | tee -a "$LOG_FILE"
          preflight_failures=$((preflight_failures + 1))
          continue
        fi
        local prior l1 cond residual refhash workhash
        prior=$(echo "$out" | grep -oE "BBSE estimate: \[[^]]*\]" | sed 's/BBSE estimate: //')
        l1=$(echo "$out" | grep -oE "l1_error=[0-9.eE+-]+" | cut -d= -f2)
        cond=$(echo "$out" | grep -oE "cond=[0-9.eE+-]+" | cut -d= -f2)
        residual=$(echo "$out" | grep -oE "residual=[0-9.eE+-]+" | cut -d= -f2)
        printf "%-22s %-6s %-6s %-32s %-10s %-8s %-10s\n" \
          "$method" "$label_budget" "$seed" "$prior" "$l1" "$cond" "$residual" | tee -a "$LOG_FILE"
      done
    done
  done

  if [ "$preflight_failures" -ne 0 ]; then
    log "=== preflight FAILED: $preflight_failures/24 dry-run cells failed ==="
    return 1
  fi

  log "=== preflight OK: tests passed, no teacher-query paths, 24/24 dry-runs OK ==="
  return 0
}

if ! preflight; then
  echo "Preflight failed; not starting GPU training. See $LOG_FILE." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# GPU grid
# ---------------------------------------------------------------------------
run_one() {
  local method="$1"
  local label_budget="$2"
  local seed="$3"
  local path
  path="$(result_path "$method" "$label_budget" "$seed")"

  if [ -e "$path" ] && cell_is_valid "$path" "$method" "$label_budget" "$seed"; then
    log "SKIP valid method=$method L=$label_budget seed=$seed"
    run_skipped=$((run_skipped + 1))
    return
  fi

  if [ -e "$path" ]; then
    quarantine_invalid "$path"
  fi

  local start_ts end_ts status
  start_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  log "START method=$method L=$label_budget seed=$seed start=$start_ts"
  run_started=$((run_started + 1))

  status=0
  if ! uv run python scripts/train_amazon3_source_prior.py \
      --method "$method" \
      --labels-per-class "$label_budget" \
      --seed "$seed" \
      --output-dir "$OUT_DIR"; then
    status=1
  fi
  end_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  if [ "$status" -ne 0 ]; then
    log "FAILED method=$method L=$label_budget seed=$seed status=$status start=$start_ts end=$end_ts"
    run_failures=$((run_failures + 1))
    return
  fi

  if ! cell_is_valid "$path" "$method" "$label_budget" "$seed"; then
    log "INVALID RESULT method=$method L=$label_budget seed=$seed path=$path start=$start_ts end=$end_ts"
    quarantine_invalid "$path"
    run_failures=$((run_failures + 1))
    return
  fi

  run_completed=$((run_completed + 1))
  log "OK method=$method L=$label_budget seed=$seed status=0 start=$start_ts end=$end_ts"
}

overall_start="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
log "=== amazon3_source_prior_transfer_v1 runner begin ==="
log "OUT_DIR=$OUT_DIR expected_cells=24 (4 methods x 2 budgets x 3 seeds)"

for label_budget in "${LABEL_BUDGETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_one "$method" "$label_budget" "$seed"
    done
  done
done

overall_end="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
log "=== runner end started=$run_started skipped=$run_skipped completed=$run_completed failures=$run_failures ==="
log "overall_start=$overall_start overall_end=$overall_end"

total_valid=$((run_skipped + run_completed))
echo "started=$run_started skipped=$run_skipped completed=$run_completed failures=$run_failures (valid cells: $total_valid/24)"

if [ "$run_failures" -ne 0 ] || [ "$total_valid" -ne 24 ]; then
  echo "Amazon-3 source-prior-transfer GPU grid incomplete; rerun this script to resume." >&2
  exit 1
fi

log "All 24 cells complete and valid under $OUT_DIR. Running the aggregate analyser."
if ! uv run python scripts/analyse_amazon3_source_prior.py 2>&1 | tee -a "$LOG_FILE"; then
  log "=== aggregate analyser FAILED ==="
  exit 1
fi

echo "All 24 cells are complete under $OUT_DIR; reports written to outputs/reports/amazon3_source_prior_transfer_v1.{json,md}."
