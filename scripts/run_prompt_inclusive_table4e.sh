#!/bin/bash
# Run the prompt-inclusive, like-for-like Table 4e student grid.

set -uo pipefail
cd "$(dirname "$0")/.."

run_failures=0
teacher_names=(
  multi-mean-guided
  multi-probmean-guided
  multi-prior-matched-guided
  multi-scale-aligned-guided
  multi-prior-scale-aligned-guided
)

run_one() {
  local dataset_name="$1"
  local label_budget="$2"
  local teacher_name="$3"
  local result_path="outputs/student/${dataset_name}_kd_${teacher_name}_L${label_budget}_s42_prompt-inclusive-v1.json"

  if [ -s "$result_path" ]; then
    echo "SKIP existing $result_path"
    return
  fi

  echo "RUN dataset=$dataset_name L=$label_budget teacher=$teacher_name"
  if ! uv run python scripts/train_student.py \
    --dataset "$dataset_name" \
    --mode kd \
    --teacher "$teacher_name" \
    --labels-per-class "$label_budget" \
    --seed 42; then
    echo "FAILED dataset=$dataset_name L=$label_budget teacher=$teacher_name" >&2
    run_failures=$((run_failures + 1))
  fi
}

for dataset_name in ag_news yahoo amazon amazon3; do
  for label_budget in 10 50; do
    for teacher_name in "${teacher_names[@]}"; do
      run_one "$dataset_name" "$label_budget" "$teacher_name"
    done
  done
done

if [ "$run_failures" -ne 0 ]; then
  echo "$run_failures Table 4e run(s) failed; rerun this script to resume." >&2
  exit 1
fi

echo "All 40 prompt-inclusive Table 4e runs are complete."
