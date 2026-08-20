#!/bin/bash
# Resumable prompt-inclusive full-baseline 3-seed extension (Phase 2).
# 4 datasets × 2 L × 3 seeds × 3 methods = 72 cells.
# Run only after Phase 1 validates. No teacher queries.

set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="outputs/student/prompt_inclusive_baselines_3seed_v1"
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
  "gemma|kd|gemma4-26b-vllm-guided"
  "mistral|kd|mistral-small-24b-vllm-guided"
  "entropy|kd|multi-entropy-guided"
)

result_path() {
  local dataset_name="$1"
  local mode="$2"
  local teacher_name="$3"
  local label_budget="$4"
  local seed="$5"
  echo "${OUT_DIR}/${dataset_name}_${mode}_${teacher_name}_L${label_budget}_s${seed}_prompt-inclusive-v1.json"
}

# Reuse the core validator with relaxed aggregation expectations for baselines.
# Gemma/Mistral/entropy cells share the same metadata contract as KD core cells
# except aggregation (entropy -> entropy_weighted; singles omit aggregation).
cell_is_valid() {
  local path="$1"
  local dataset_name="$2"
  local mode="$3"
  local teacher_name="$4"
  local label_budget="$5"
  local seed="$6"
  uv run python - "$path" "$dataset_name" "$mode" "$teacher_name" "$label_budget" "$seed" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
dataset_name = sys.argv[2]
mode = sys.argv[3]
teacher_name = sys.argv[4]
label_budget = int(sys.argv[5])
seed = int(sys.argv[6])
expected_n = {"ag_news": 7600, "yahoo": 10000, "amazon": 10000, "amazon3": 6000}[dataset_name]
expected_agg = {
    "multi-entropy-guided": "entropy_weighted",
}.get(teacher_name)

if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
checks = [
    data.get("dataset") == dataset_name,
    data.get("mode") == mode,
    data.get("teacher") == teacher_name,
    data.get("labels_per_class") == label_budget,
    data.get("seed") == seed,
    data.get("label_accounting", {}).get("policy") == "prompt-inclusive-v1",
    data.get("dataset_identity", {}).get("artifact_version") == "large-test-v1",
]
metrics = data.get("metrics", {})
for key in ("accuracy", "macro_f1"):
    value = metrics.get(key)
    checks.append(
        isinstance(value, (int, float))
        and math.isfinite(value)
        and 0.0 <= float(value) <= 1.0
    )
paired = data.get("paired_test_data", {})
checks.append(len(paired.get("predictions", [])) == expected_n)
checks.append(len(paired.get("labels", [])) == expected_n)
checks.append(len(paired.get("sample_indices", [])) == expected_n)
cfg = data.get("config", {})
required = {
    "model_name": "answerdotai/ModernBERT-base",
    "epochs": 3,
    "alpha": 0.5,
    "temperature": 2.0,
    "lr": 2e-5,
    "batch_size": 16,
    "max_length": 256,
}
checks.extend(cfg.get(k) == v for k, v in required.items())
if expected_agg is not None:
    checks.append(data.get("aggregation") == expected_agg)
raise SystemExit(0 if all(checks) else 1)
PY
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
  log "START dataset=$dataset_name method=$method_key teacher=$teacher_name L=$label_budget seed=$seed"
  run_started=$((run_started + 1))

  status=0
  if ! uv run python scripts/train_student.py \
    --dataset "$dataset_name" \
    --mode kd \
    --teacher "$teacher_name" \
    --labels-per-class "$label_budget" \
    --seed "$seed" \
    --epochs 3 \
    --alpha 0.5 \
    --temperature 2.0 \
    --lr 2e-5 \
    --batch-size 16 \
    --max-length 256 \
    --output-dir "$OUT_DIR"; then
    status=1
  fi
  end_ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  if [ "$status" -ne 0 ]; then
    log "FAILED dataset=$dataset_name method=$method_key L=$label_budget seed=$seed status=$status start=$start_ts end=$end_ts"
    run_failures=$((run_failures + 1))
    return
  fi

  if ! cell_is_valid "$path" "$dataset_name" "$mode" "$teacher_name" "$label_budget" "$seed"; then
    log "INVALID RESULT $path"
    quarantine_invalid "$path"
    run_failures=$((run_failures + 1))
    return
  fi

  log "OK dataset=$dataset_name method=$method_key L=$label_budget seed=$seed status=0 start=$start_ts end=$end_ts"
}

log "=== prompt_inclusive_baselines_3seed_v1 runner begin ==="

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
  echo "$run_failures baseline run(s) failed; rerun this script to resume." >&2
  exit 1
fi

echo "All 72 prompt-inclusive baseline cells are complete under $OUT_DIR."
