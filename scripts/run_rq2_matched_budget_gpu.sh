#!/bin/bash
# Resumable RQ2 matched-budget student grid at q/N=1.4.
# Uniform allocation vs entropy cascade, prompt-inclusive, large-test-v1.

set -uo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="outputs/student/rq2_matched_q1p4_v1"
DATASETS=(ag_news yahoo amazon amazon3)
LABEL_BUDGETS=(10 50)
SEEDS=(42 123 456 789 2024)
TARGET_QUERIES=1.4
run_failures=0

mkdir -p "$OUT_DIR"

expected_test_size() {
  case "$1" in
    ag_news) echo 7600 ;;
    yahoo) echo 10000 ;;
    amazon) echo 10000 ;;
    amazon3) echo 6000 ;;
    *) echo 0 ;;
  esac
}

result_path() {
  local dataset_name="$1"
  local teacher_name="$2"
  local label_budget="$3"
  local seed="$4"
  if [ "$teacher_name" = "multi-uniform-budget-guided" ]; then
    echo "${OUT_DIR}/${dataset_name}_kd_${teacher_name}_q1p4_L${label_budget}_s${seed}_a${seed}_prompt-inclusive-v1.json"
  else
    echo "${OUT_DIR}/${dataset_name}_kd_${teacher_name}_q1p4_L${label_budget}_s${seed}_prompt-inclusive-v1.json"
  fi
}

cell_is_valid() {
  local path="$1"
  local dataset_name="$2"
  local teacher_name="$3"
  local label_budget="$4"
  local seed="$5"
  local expected_n
  expected_n="$(expected_test_size "$dataset_name")"

  uv run python - "$path" "$dataset_name" "$teacher_name" "$label_budget" "$seed" "$expected_n" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
dataset_name = sys.argv[2]
teacher_name = sys.argv[3]
label_budget = int(sys.argv[4])
seed = int(sys.argv[5])
expected_n = int(sys.argv[6])

if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit(1)

data = json.loads(path.read_text(encoding="utf-8"))
checks = [
    data.get("dataset") == dataset_name,
    data.get("teacher") == teacher_name,
    data.get("labels_per_class") == label_budget,
    data.get("seed") == seed,
    data.get("label_accounting", {}).get("policy") == "prompt-inclusive-v1",
    data.get("dataset_identity", {}).get("artifact_version") == "large-test-v1",
    abs(float(data.get("queries_per_sample", -1.0)) - 1.4) < 1e-3,
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
preds = paired.get("predictions", [])
labels = paired.get("labels", [])
checks.append(len(preds) == expected_n)
checks.append(len(labels) == expected_n)
checks.append(len(paired.get("sample_indices", [])) == expected_n)

if teacher_name == "multi-uniform-budget-guided":
    policy = data.get("budget_policy", {})
    checks.append(policy.get("allocation_seed") == seed)
    checks.append(abs(float(policy.get("realised_queries_per_sample", -1.0)) - 1.4) < 1e-3)
    checks.append(isinstance(policy.get("selected_count"), int))

raise SystemExit(0 if all(checks) else 1)
PY
}

run_one() {
  local dataset_name="$1"
  local teacher_name="$2"
  local label_budget="$3"
  local seed="$4"
  local path
  path="$(result_path "$dataset_name" "$teacher_name" "$label_budget" "$seed")"

  if cell_is_valid "$path" "$dataset_name" "$teacher_name" "$label_budget" "$seed"; then
    echo "SKIP valid $path"
    return
  fi

  if [ -e "$path" ]; then
    echo "RESTART invalid/incomplete $path"
  else
    echo "RUN dataset=$dataset_name teacher=$teacher_name L=$label_budget seed=$seed"
  fi

  local cmd=(
    uv run python scripts/train_student.py
    --dataset "$dataset_name"
    --mode kd
    --teacher "$teacher_name"
    --labels-per-class "$label_budget"
    --seed "$seed"
    --epochs 3
    --alpha 0.5
    --temperature 2.0
    --lr 2e-5
    --batch-size 16
    --max-length 256
    --output-dir "$OUT_DIR"
  )

  if [ "$teacher_name" = "multi-cascade-guided" ]; then
    cmd+=(--cascade-target-queries "$TARGET_QUERIES" --cascade-order qwen-first)
  else
    cmd+=(--uniform-target-queries "$TARGET_QUERIES" --allocation-seed "$seed")
  fi

  if ! "${cmd[@]}"; then
    echo "FAILED dataset=$dataset_name teacher=$teacher_name L=$label_budget seed=$seed" >&2
    run_failures=$((run_failures + 1))
    return
  fi

  if ! cell_is_valid "$path" "$dataset_name" "$teacher_name" "$label_budget" "$seed"; then
    echo "INVALID RESULT $path" >&2
    run_failures=$((run_failures + 1))
  fi
}

for dataset_name in "${DATASETS[@]}"; do
  for label_budget in "${LABEL_BUDGETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for teacher_name in multi-cascade-guided multi-uniform-budget-guided; do
        run_one "$dataset_name" "$teacher_name" "$label_budget" "$seed"
      done
    done
  done
done

if [ "$run_failures" -ne 0 ]; then
  echo "$run_failures RQ2 matched-budget run(s) failed; rerun this script to resume." >&2
  exit 1
fi

echo "All 80 RQ2 matched-budget cells are complete under $OUT_DIR."
