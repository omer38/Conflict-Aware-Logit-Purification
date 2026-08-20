#!/bin/bash
# Resumable student-level A.12 prior-misspecification grid.
#
# Balanced-pool, assumed-prior sensitivity check: only the prior supplied to
# multi-prior-scale-aligned-guided matching changes across conditions. The
# actual training pool and the fixed balanced test set are unchanged in every
# condition. This trains only the 36 new mild/strong cells; the 18 existing
# correct-prior cells in outputs/student/prompt_inclusive_core_3seed_v1/ are
# validated but never retrained (unless missing or invalid).
#
# 3 datasets x 2 label budgets x 3 seeds x 2 misspecified conditions = 36 new
# GPU runs. Combined with the 18 pre-existing correct-prior cells, the full
# comparison grid is 54 cells. No teacher queries (cached logits only).

set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="outputs/student/prior_misspecification_a12_v1"
BASELINE_DIR="outputs/student/prompt_inclusive_core_3seed_v1"
LOG_FILE="${OUT_DIR}/run_log.txt"
INVALID_DIR="${OUT_DIR}/_invalid"
DATASETS=(ag_news amazon yahoo)
LABEL_BUDGETS=(10 50)
SEEDS=(42 123 456)
CONDITIONS=(mild strong)
DOMINANT_CLASS=0
TEACHER="multi-prior-scale-aligned-guided"

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

# Mirrors the single naming convention shared by train_student.py and
# scripts/analyse_student_prior_misspecification.py.
result_path() {
  local dataset_name="$1"
  local condition="$2"
  local label_budget="$3"
  local seed="$4"
  local prior_tag=""
  if [ "$condition" != "correct" ]; then
    prior_tag="_prior-${condition}-d${DOMINANT_CLASS}"
  fi
  echo "${OUT_DIR}/${dataset_name}_kd_${TEACHER}${prior_tag}_L${label_budget}_s${seed}_prompt-inclusive-v1.json"
}

baseline_result_path() {
  local dataset_name="$1"
  local label_budget="$2"
  local seed="$3"
  echo "${BASELINE_DIR}/${dataset_name}_kd_${TEACHER}_L${label_budget}_s${seed}_prompt-inclusive-v1.json"
}

cell_is_valid() {
  local path="$1"
  local dataset_name="$2"
  local condition="$3"
  local label_budget="$4"
  local seed="$5"
  uv run python scripts/analyse_student_prior_misspecification.py \
    --check-cell "$path" \
    --dataset "$dataset_name" \
    --condition "$condition" \
    --labels-per-class "$label_budget" \
    --seed "$seed" \
    --dominant-class "$DOMINANT_CLASS" >/dev/null 2>&1
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

run_one() {
  local dataset_name="$1"
  local label_budget="$2"
  local seed="$3"
  local condition="$4"
  local path
  path="$(result_path "$dataset_name" "$condition" "$label_budget" "$seed")"

  if [ -e "$path" ] && cell_is_valid "$path" "$dataset_name" "$condition" "$label_budget" "$seed"; then
    log "SKIP valid dataset=$dataset_name condition=$condition L=$label_budget seed=$seed"
    run_skipped=$((run_skipped + 1))
    return
  fi

  if [ -e "$path" ]; then
    quarantine_invalid "$path"
  fi

  local start_ts end_ts status
  start_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  log "START dataset=$dataset_name condition=$condition L=$label_budget seed=$seed start=$start_ts"
  run_started=$((run_started + 1))

  local cmd=(
    uv run python scripts/train_student.py
    --dataset "$dataset_name"
    --mode kd
    --teacher "$TEACHER"
    --prior-condition "$condition"
    --prior-dominant-class "$DOMINANT_CLASS"
    --labels-per-class "$label_budget"
    --seed "$seed"
    --epochs 3
    --temperature 2.0
    --alpha 0.5
    --lr 2e-5
    --batch-size 16
    --max-length 256
    --output-dir "$OUT_DIR"
  )

  status=0
  if ! "${cmd[@]}"; then
    status=1
  fi
  end_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  if [ "$status" -ne 0 ]; then
    log "FAILED dataset=$dataset_name condition=$condition L=$label_budget seed=$seed status=$status start=$start_ts end=$end_ts"
    run_failures=$((run_failures + 1))
    return
  fi

  if ! cell_is_valid "$path" "$dataset_name" "$condition" "$label_budget" "$seed"; then
    log "INVALID RESULT dataset=$dataset_name condition=$condition L=$label_budget seed=$seed path=$path start=$start_ts end=$end_ts"
    quarantine_invalid "$path"
    run_failures=$((run_failures + 1))
    return
  fi

  run_completed=$((run_completed + 1))
  log "OK dataset=$dataset_name condition=$condition L=$label_budget seed=$seed status=0 start=$start_ts end=$end_ts"
}

overall_start="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
log "=== prior_misspecification_a12_v1 runner begin ==="
log "OUT_DIR=$OUT_DIR expected_new_cells=36 expected_correct_baselines=18 expected_total_comparison_cells=54"

# The 18 correct-prior cells are never retrained here; only validated. A
# missing/invalid baseline fails the run clearly rather than silently
# retraining or substituting a different protocol.
baseline_failures=0
for label_budget in "${LABEL_BUDGETS[@]}"; do
  for dataset_name in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      bpath="$(baseline_result_path "$dataset_name" "$label_budget" "$seed")"
      if [ -e "$bpath" ] && cell_is_valid "$bpath" "$dataset_name" "correct" "$label_budget" "$seed"; then
        log "BASELINE OK dataset=$dataset_name L=$label_budget seed=$seed path=$bpath"
      else
        log "BASELINE INVALID/MISSING dataset=$dataset_name L=$label_budget seed=$seed path=$bpath"
        baseline_failures=$((baseline_failures + 1))
      fi
    done
  done
done

if [ "$baseline_failures" -ne 0 ]; then
  log "=== runner aborting: $baseline_failures/18 correct-prior baseline cells missing or invalid ==="
  echo "$baseline_failures correct-prior baseline cell(s) missing or invalid under $BASELINE_DIR." >&2
  echo "This runner does not retrain baselines; fix or regenerate them first." >&2
  exit 1
fi
log "All 18 correct-prior baseline cells validated."

for label_budget in "${LABEL_BUDGETS[@]}"; do
  for dataset_name in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for condition in "${CONDITIONS[@]}"; do
        run_one "$dataset_name" "$label_budget" "$seed" "$condition"
      done
    done
  done
done

overall_end="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
log "=== runner end started=$run_started skipped=$run_skipped completed=$run_completed failures=$run_failures ==="
log "overall_start=$overall_start overall_end=$overall_end"

total_valid=$((run_skipped + run_completed))
echo "started=$run_started skipped=$run_skipped completed=$run_completed failures=$run_failures (valid new cells: $total_valid/36)"

if [ "$run_failures" -ne 0 ] || [ "$total_valid" -ne 36 ]; then
  echo "Prior-misspecification GPU grid incomplete; rerun this script to resume." >&2
  exit 1
fi

echo "All 36 new prior-misspecification cells are complete under $OUT_DIR."
echo "Combined with the 18 correct-prior baselines, all 54 comparison cells are ready for analysis."
