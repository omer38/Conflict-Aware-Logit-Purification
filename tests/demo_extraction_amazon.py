#!/usr/bin/env python
"""Teaching demo (Amazon): raw token output -> k-class vector.

For a few Amazon review samples this script prints, side by side:

  1. The RAW top-20 tokens the LLM returned for the first output position
     (its distribution over *vocabulary*), marking which survive parsing.
  2. The CLEAN classes that actually appeared in top-k (not at LOGPROB_FLOOR),
     sorted by logit descending.

Run:
    uv run python tests/demo_extraction_amazon.py
    uv run python tests/demo_extraction_amazon.py --teacher gemma4-26b-mlx
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.teachers.llm import TEACHER_PRESETS, get_client, complete, has_api_credentials
from src.teachers.extraction import (
    build_prompt,
    build_fewshot_pool,
    fewshot_examples_for_index,
    make_verbalizers,
    parse_logprobs_to_logits,
)

N_SAMPLES = 4
DEFAULT_TEACHER = "gpt-4.1-mini"


def load_env() -> None:
    """Load .env into os.environ if it exists (so OPENAI_API_KEY is picked up)."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def demo_sample(text: str, true_label: int, label_names: list[str], response) -> None:
    num_classes = len(label_names)
    verbalizers = make_verbalizers(num_classes)
    token_to_class = {v: i for i, v in enumerate(verbalizers)}

    print(f'  Text       : "{text[:90]}{"..." if len(text) > 90 else ""}"')
    print(f"  True label : {true_label} = {label_names[true_label]}")
    print(f'  LLM said   : "{response.choices[0].message.content.strip()}"')

    # 1. The RAW vocabulary distribution for the first output token.
    top = response.choices[0].logprobs.content[0].top_logprobs
    print(f"\n  RAW top-{len(top)} tokens the LLM returned (its view of the whole vocab):")
    print(f"    {'token':>10} {'logprob':>9} {'prob':>8}   kept?")
    for entry in top:
        prob = math.exp(entry.logprob)
        stripped = entry.token.strip()
        if stripped in token_to_class:
            idx = token_to_class[stripped]
            verdict = f"KEEP -> class {idx} ({label_names[idx]})"
        else:
            verdict = "skip (not a class letter)"
        print(f"    {entry.token!r:>10} {entry.logprob:>9.3f} {prob:>8.2%}   {verdict}")

    # 2. The CLEAN k-class vector -- show only classes that appeared in top-k.
    logits, floored = parse_logprobs_to_logits(response, verbalizers)
    observed = [(i, logits[i]) for i in range(num_classes) if not floored[i]]
    observed.sort(key=lambda x: x[1], reverse=True)

    floor_count = num_classes - len(observed)
    print(
        f"\n  CLEAN output: {len(observed)} classes observed in top-k "
        f"({floor_count}/{num_classes} floored -- not seen by the model):"
    )
    print(f"    {'class':>5}  {'name':<35} {'logit':>8}  {'prob':>8}")

    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    for i, logit in observed:
        print(f"    {i:>5}  {label_names[i]:<35} {logit:>8.3f}  {probs[i]:>8.2%}")

    pred = int(logits.argmax())
    mark = "correct" if pred == true_label else "WRONG"
    print(f"\n  Prediction : class {pred} = {label_names[pred]}  ({mark})")
    print("-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo logit extraction on Amazon samples.")
    parser.add_argument(
        "--teacher",
        default=DEFAULT_TEACHER,
        choices=list(TEACHER_PRESETS.keys()),
        help="Teacher model preset name.",
    )
    args = parser.parse_args()

    load_env()
    config = TEACHER_PRESETS[args.teacher]
    if not has_api_credentials(config):
        print(f"{config.api_key_env} not set. Set it (or put it in .env) to run this demo.")
        sys.exit(0)

    amazon = load_dataset_by_name("amazon")
    client = get_client(config)
    verbalizers = make_verbalizers(amazon.num_classes)
    fewshot_pool = build_fewshot_pool(amazon)

    print(f"Dataset: Amazon  |  {amazon.num_classes} star-rating classes")
    print(f"Teacher: {config.name} (model={config.model})")
    print(f"Note: top_logprobs={config.max_top_logprobs}\n")
    print("=" * 70)

    for i in range(N_SAMPLES):
        sample = amazon.train[i]
        text = sample[amazon.text_column]
        true_label = sample[amazon.label_column]

        messages = build_prompt(
            text,
            amazon.label_names,
            verbalizers,
            fewshot_examples_for_index(amazon, verbalizers, fewshot_pool, i),
        )
        response = complete(
            client,
            config,
            messages,
            guided_choices=verbalizers if config.constrain_to_verbalizers else None,
        )

        print(f"SAMPLE {i}")
        demo_sample(text, true_label, amazon.label_names, response)
        print()


if __name__ == "__main__":
    main()
