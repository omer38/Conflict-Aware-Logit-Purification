#!/usr/bin/env bash
# Run supervised + single-teacher KD grid for one or more datasets.
set -uo pipefail

DATASETS=("${@:-yahoo amazon}")
TEACHERS=(
  "qwen3.5-27b-vllm-guided"
  "gemma4-26b-vllm-guided"
  "mistral-small-24b-vllm-guided"
)
L_VALUES=(10 50 500)
SEED=42

run_train() {
  local extra_args=("$@")
  if ! uv run python scripts/train_student.py "${extra_args[@]}"; then
    echo "FAILED: ${extra_args[*]}"
    uv run python -c "import torch; torch.mps.empty_cache() if torch.backends.mps.is_available() else None" 2>/dev/null || true
    return 1
  fi
  uv run python -c "import torch; torch.mps.empty_cache() if torch.backends.mps.is_available() else None" 2>/dev/null || true
}

batch_and_maxlen() {
  local ds=$1
  local l=$2
  local mode=$3
  if [[ "${l}" -eq 500 ]]; then
    echo "4 128"
  elif [[ "${ds}" == "yahoo" ]] || [[ "${ds}" == "amazon" && "${mode}" == "kd" ]]; then
    echo "8 128"
  else
    echo "16 256"
  fi
}

for DS in "${DATASETS[@]}"; do
  echo "######## DATASET: ${DS} ########"
  for L in "${L_VALUES[@]}"; do
    OUT="outputs/student/${DS}_supervised_none_L${L}_s${SEED}.json"
    read -r BATCH MAXLEN <<< "$(batch_and_maxlen "${DS}" "${L}" supervised)"
    if [[ -f "${OUT}" ]]; then
      echo "===== skip ${DS} supervised L=${L} (exists) ====="
    else
      echo "===== ${DS} supervised L=${L} batch=${BATCH} max_length=${MAXLEN} ====="
      run_train \
        --dataset "${DS}" --mode supervised \
        --labels-per-class "${L}" --seed "${SEED}" \
        --batch-size "${BATCH}" --max-length "${MAXLEN}" || true
    fi
  done

  for TEACHER in "${TEACHERS[@]}"; do
    for L in "${L_VALUES[@]}"; do
      OUT="outputs/student/${DS}_kd_${TEACHER}_L${L}_s${SEED}.json"
      read -r BATCH MAXLEN <<< "$(batch_and_maxlen "${DS}" "${L}" kd)"
      if [[ -f "${OUT}" ]]; then
        echo "===== skip ${DS} kd ${TEACHER} L=${L} (exists) ====="
      else
        echo "===== ${DS} kd ${TEACHER} L=${L} batch=${BATCH} max_length=${MAXLEN} ====="
        run_train \
          --dataset "${DS}" --mode kd --teacher "${TEACHER}" \
          --labels-per-class "${L}" --seed "${SEED}" \
          --batch-size "${BATCH}" --max-length "${MAXLEN}" || true
      fi
    done
  done

  for L in "${L_VALUES[@]}"; do
    OUT="outputs/student/${DS}_kd_multi-mean-guided_L${L}_s${SEED}.json"
    read -r BATCH MAXLEN <<< "$(batch_and_maxlen "${DS}" "${L}" kd)"
    if [[ -f "${OUT}" ]]; then
      echo "===== skip ${DS} kd multi-mean-guided L=${L} (exists) ====="
    else
      echo "===== ${DS} kd multi-mean-guided L=${L} batch=${BATCH} max_length=${MAXLEN} ====="
      run_train \
        --dataset "${DS}" --mode kd --teacher multi-mean-guided \
        --labels-per-class "${L}" --seed "${SEED}" \
        --batch-size "${BATCH}" --max-length "${MAXLEN}" || true
    fi
  done

  for L in "${L_VALUES[@]}"; do
    OUT="outputs/student/${DS}_kd_multi-entropy-guided_L${L}_s${SEED}.json"
    read -r BATCH MAXLEN <<< "$(batch_and_maxlen "${DS}" "${L}" kd)"
    if [[ -f "${OUT}" ]]; then
      echo "===== skip ${DS} kd multi-entropy-guided L=${L} (exists) ====="
    else
      echo "===== ${DS} kd multi-entropy-guided L=${L} batch=${BATCH} max_length=${MAXLEN} ====="
      run_train \
        --dataset "${DS}" --mode kd --teacher multi-entropy-guided \
        --labels-per-class "${L}" --seed "${SEED}" \
        --batch-size "${BATCH}" --max-length "${MAXLEN}" || true
    fi
  done
done

echo "######## DONE ########"
