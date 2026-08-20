#!/bin/bash
# Runs the GPU experiments identified as missing from docs/progress_report5_draft.md:
# hygiene reruns (checklist items 1-2) + pending ablations (i)-(iii) from §7.
set -uo pipefail
cd "$(dirname "$0")/.."

run() {
  echo "=== START $* $(date) ==="
  uv run python scripts/train_student.py "$@"
  echo "=== END $* $(date) (exit=$?) ==="
}

# --- Hygiene: Yahoo supervised + Yahoo/Amazon Gemma/Mistral/entropy singles, matched config (14 runs) ---
for L in 10 50; do
  run --dataset yahoo --mode supervised --labels-per-class $L --seed 42
  run --dataset yahoo --mode kd --teacher gemma4-26b-vllm-guided --labels-per-class $L --seed 42
  run --dataset yahoo --mode kd --teacher mistral-small-24b-vllm-guided --labels-per-class $L --seed 42
  run --dataset yahoo --mode kd --teacher multi-entropy-guided --labels-per-class $L --seed 42
  run --dataset amazon --mode kd --teacher gemma4-26b-vllm-guided --labels-per-class $L --seed 42
  run --dataset amazon --mode kd --teacher mistral-small-24b-vllm-guided --labels-per-class $L --seed 42
  run --dataset amazon --mode kd --teacher multi-entropy-guided --labels-per-class $L --seed 42
done

# --- Ablation (i): prior-matching-only vs scale-alignment-only (12 runs) ---
for L in 10 50; do
  for ds in ag_news yahoo amazon; do
    run --dataset $ds --mode kd --teacher multi-prior-matched-guided --labels-per-class $L --seed 42
    run --dataset $ds --mode kd --teacher multi-scale-aligned-guided --labels-per-class $L --seed 42
  done
done

# --- Ablation (ii): prior-matched Qwen as the fair 1.0 q/N baseline (8 runs) ---
for L in 10 50; do
  for ds in ag_news yahoo amazon amazon3; do
    run --dataset $ds --mode kd --teacher qwen-prior-matched-guided --labels-per-class $L --seed 42
  done
done

# --- Ablation (iii): cascade teacher-order sensitivity on AG News (4 runs) ---
for L in 10 50; do
  run --dataset ag_news --mode kd --teacher multi-cascade-guided --cascade-order gemma-first --labels-per-class $L --seed 42
  run --dataset ag_news --mode kd --teacher multi-cascade-guided --cascade-order mistral-first --labels-per-class $L --seed 42
done

echo "=== ALL MISSING EXPERIMENTS DONE $(date) ==="
