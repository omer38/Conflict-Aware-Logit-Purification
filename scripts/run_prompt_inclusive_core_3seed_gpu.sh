#!/bin/bash
# Resumable prompt-inclusive core 3-seed student grid (mandatory Phase 1).
# 4 datasets × 2 L × 3 seeds × 8 methods = 192 cells.
# No teacher queries. Isolated output directory.

set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="outputs/student/prompt_inclusive_core_3seed_v1"
LOG_FILE="${OUT_DIR}/run_log.txt"
INVALID_DIR="${OUT_DIR}/_invalid"
DATASETS=(ag_news yahoo amazon amazon3)
LABEL_BUDGETS=(10 50)
SEEDS=(42 123 456)
run_failures=0
run_started=0
run_skipped=0

mkdir -p "$OUT_DIR" "$INVALID_DIR"

log() {
  local msg="$1"
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[$ts] $msg" | tee -a "$LOG_FILE"
}

# method_key|mode|teacher
METHODS=(
  "supervised|supervised|none"
  "raw_qwen|kd|qwen3.5-27b-vllm-guided"
  "prior_matched_qwen|kd|qwen-prior-matched-guided"
  "logit_mean|kd|multi-mean-guided"
  "prob_mean|kd|multi-probmean-guided"
  "prior_matched|kd|multi-prior-matched-guided"
  "scale_aligned|kd|multi-scale-aligned-guided"
  "combined|kd|multi-prior-scale-aligned-guided"
)

result_path() {
  local dataset_name="$1"
  local mode="$2"
  local teacher_name="$3"
  local label_budget="$4"
  local seed="$5"
  echo "${OUT_DIR}/${dataset_name}_${mode}_${teacher_name}_L${label_budget}_s${seed}_prompt-inclusive-v1.json"
}

cell_is_valid() {
  local path="$1"
  local dataset_name="$2"
  local mode="$3"
  local teacher_name="$4"
  local label_budget="$5"
  local seed="$6"
  uv run python scripts/analyse_prompt_inclusive_core_students.py \
    --check-cell "$path" \
    --dataset "$dataset_name" \
    --mode "$mode" \
    --teacher "$teacher_name" \
    --labels-per-class "$label_budget" \
    --seed "$seed" >/dev/null 2>&1
}

quarantine_invalid() {
  local path="$1"
  if [ ! -e "$path" ]; then
    return
  fi
  local base
  local stamp
  base="$(basename "$path")"
  stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  local dest="${INVALID_DIR}/${stamp}_${base}"
  mv "$path" "$dest"
  log "QUARANTINE invalid -> $dest"
}

run_one() {
  local dataset_name="$1"
  local mode="$2"
  local teacher_name="$3"
  local label_budget="$4"
  local seed="$5"
  local method_key="$6"
  local path
  path="$(result_path "$dataset_name" "$mode" "$teacher_name" "$label_budget" "$seed")"

  if cell_is_valid "$path" "$dataset_name" "$mode" "$teacher_name" "$label_budget" "$seed"; then
    log "SKIP valid dataset=$dataset_name method=$method_key L=$label_budget seed=$seed"
    run_skipped=$((run_skipped + 1))
    return
  fi

  if [ -e "$path" ]; then
    quarantine_invalid "$path"
  fi

  local start_ts end_ts status
  start_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  log "START dataset=$dataset_name method=$method_key mode=$mode teacher=$teacher_name L=$label_budget seed=$seed"
  run_started=$((run_started + 1))

  local cmd=(
    uv run python scripts/train_student.py
    --dataset "$dataset_name"
    --mode "$mode"
    --labels-per-class "$label_budget"
    --seed "$seed"
    --epochs 3
    --temperature 2.0
    --lr 2e-5
    --batch-size 16
    --max-length 256
    --output-dir "$OUT_DIR"
  )

  if [ "$mode" = "kd" ]; then
    cmd+=(--teacher "$teacher_name" --alpha 0.5)
  fi

  status=0
  if ! "${cmd[@]}"; then
    status=1
  fi
  end_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  if [ "$status" -ne 0 ]; then
    log "FAILED dataset=$dataset_name method=$method_key L=$label_budget seed=$seed status=$status start=$start_ts end=$end_ts"
    run_failures=$((run_failures + 1))
    return
  fi

  if ! cell_is_valid "$path" "$dataset_name" "$mode" "$teacher_name" "$label_budget" "$seed"; then
    log "INVALID RESULT dataset=$dataset_name method=$method_key L=$label_budget seed=$seed path=$path start=$start_ts end=$end_ts"
    quarantine_invalid "$path"
    run_failures=$((run_failures + 1))
    return
  fi

  log "OK dataset=$dataset_name method=$method_key L=$label_budget seed=$seed status=0 start=$start_ts end=$end_ts"
}

log "=== prompt_inclusive_core_3seed_v1 runner begin ==="
log "OUT_DIR=$OUT_DIR expected_cells=192"

for label_budget in "${LABEL_BUDGETS[@]}"; do
  for dataset_name in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for method_spec in "${METHODS[@]}"; do
        IFS='|' read -r method_key mode teacher_name <<<"$method_spec"
        run_one "$dataset_name" "$mode" "$teacher_name" "$label_budget" "$seed" "$method_key"
      done
    done
  done
done

log "=== runner end failures=$run_failures started=$run_started skipped=$run_skipped ==="

if [ "$run_failures" -ne 0 ]; then
  echo "$run_failures prompt-inclusive core run(s) failed; rerun this script to resume." >&2
  exit 1
fi

echo "All 192 prompt-inclusive core cells are complete under $OUT_DIR."
