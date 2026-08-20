#!/bin/bash
# Resumable 48-cell RQ2 depth-matched random control.

set -uo pipefail
cd "$(dirname "$0")/.."

TEACHER="multi-depth-matched-random-guided"
DATASETS=(ag_news yahoo amazon amazon3)
LABEL_BUDGETS=(10 50)
SEEDS=(42 123 456)
QUERY_RATES=(1.4 2.0)
run_failures=0
run_started=0
run_skipped=0

output_dir() {
  if [ "$1" = "1.4" ]; then
    echo "outputs/student/rq2_matched_q1p4_v1"
  else
    echo "outputs/student/rq2_matched_q2_3seed_v1"
  fi
}

query_tag() {
  if [ "$1" = "1.4" ]; then
    echo "q1p4"
  else
    echo "q2"
  fi
}

log() {
  local out_dir="$1"
  shift
  local ts
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "[$ts] $*" | tee -a "${out_dir}/depth_matched_random_run_log.txt"
}

result_path() {
  local query_rate="$1"
  local dataset_name="$2"
  local label_budget="$3"
  local seed="$4"
  local out_dir tag
  out_dir="$(output_dir "$query_rate")"
  tag="$(query_tag "$query_rate")"
  echo "${out_dir}/${dataset_name}_kd_${TEACHER}_${tag}_L${label_budget}_s${seed}_a${seed}_prompt-inclusive-v1.json"
}

cell_is_valid() {
  local path="$1"
  local query_rate="$2"
  local dataset_name="$3"
  local label_budget="$4"
  local seed="$5"
  uv run python scripts/analyse_rq2_depth_matched_random.py \
    --check-cell "$path" \
    --dataset "$dataset_name" \
    --labels-per-class "$label_budget" \
    --seed "$seed" \
    --target-queries "$query_rate"
}

quarantine_invalid() {
  local path="$1"
  local out_dir="$2"
  if [ ! -e "$path" ]; then
    return
  fi
  local invalid_dir base stamp destination
  invalid_dir="${out_dir}/_invalid_depth_matched_random"
  mkdir -p "$invalid_dir"
  base="$(basename "$path")"
  stamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  destination="${invalid_dir}/${stamp}_${base}"
  mv "$path" "$destination"
  log "$out_dir" "QUARANTINE invalid -> $destination"
}

run_one() {
  local query_rate="$1"
  local dataset_name="$2"
  local label_budget="$3"
  local seed="$4"
  local out_dir path
  out_dir="$(output_dir "$query_rate")"
  path="$(result_path "$query_rate" "$dataset_name" "$label_budget" "$seed")"
  mkdir -p "$out_dir" "${out_dir}/_invalid_depth_matched_random"

  if cell_is_valid "$path" "$query_rate" "$dataset_name" "$label_budget" "$seed"; then
    log "$out_dir" "SKIP valid $path"
    run_skipped=$((run_skipped + 1))
    return
  fi
  if [ -e "$path" ]; then
    quarantine_invalid "$path" "$out_dir"
  fi

  local start_ts end_ts status
  start_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  log "$out_dir" "START q=$query_rate dataset=$dataset_name L=$label_budget seed=$seed"
  run_started=$((run_started + 1))

  status=0
  if ! uv run python scripts/train_student.py \
    --dataset "$dataset_name" \
    --mode kd \
    --teacher "$TEACHER" \
    --depth-matched-target-queries "$query_rate" \
    --allocation-seed "$seed" \
    --labels-per-class "$label_budget" \
    --seed "$seed" \
    --model-name answerdotai/ModernBERT-base \
    --epochs 3 \
    --alpha 0.5 \
    --temperature 2.0 \
    --lr 2e-5 \
    --batch-size 16 \
    --max-length 256 \
    --output-dir "$out_dir"; then
    status=1
  fi
  end_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  if [ "$status" -ne 0 ]; then
    log "$out_dir" "FAILED q=$query_rate dataset=$dataset_name L=$label_budget seed=$seed status=$status start=$start_ts end=$end_ts"
    run_failures=$((run_failures + 1))
    return
  fi
  if ! cell_is_valid "$path" "$query_rate" "$dataset_name" "$label_budget" "$seed"; then
    log "$out_dir" "INVALID q=$query_rate dataset=$dataset_name L=$label_budget seed=$seed start=$start_ts end=$end_ts"
    quarantine_invalid "$path" "$out_dir"
    run_failures=$((run_failures + 1))
    return
  fi
  log "$out_dir" "OK q=$query_rate dataset=$dataset_name L=$label_budget seed=$seed status=0 start=$start_ts end=$end_ts"
}

SMOKE_OUT="$(output_dir 1.4)"
mkdir -p "$SMOKE_OUT"
log "$SMOKE_OUT" "=== depth-matched random smoke test begin ==="
run_one 1.4 ag_news 10 42
if [ "$run_failures" -ne 0 ]; then
  log "$SMOKE_OUT" "Smoke test failed; full grid not started."
  exit 1
fi
log "$SMOKE_OUT" "=== smoke test validated; full grid begin ==="

for query_rate in "${QUERY_RATES[@]}"; do
  for dataset_name in "${DATASETS[@]}"; do
    for label_budget in "${LABEL_BUDGETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_one "$query_rate" "$dataset_name" "$label_budget" "$seed"
      done
    done
  done
done

for query_rate in "${QUERY_RATES[@]}"; do
  log "$(output_dir "$query_rate")" "=== runner end failures=$run_failures started=$run_started skipped=$run_skipped ==="
done

if [ "$run_failures" -ne 0 ]; then
  echo "$run_failures depth-matched random run(s) failed; rerun to resume." >&2
  exit 1
fi

echo "All 48 depth-matched random RQ2 cells are complete and validated."
