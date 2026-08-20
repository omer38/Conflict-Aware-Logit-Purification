#!/usr/bin/env python
"""Teaching demo: watch how an LLM's raw token output becomes a k-class vector.

For a few AG News samples this script prints, side by side:

  1. The RAW top-20 tokens the LLM returned for the first output position
     (its distribution over *vocabulary*), marking which survive parsing.
  2. The CLEAN 4-class logit/probability vector that comes out the other end.

This makes the "soft constraint vs hard constraint" idea concrete:
  - SOFT (the prompt): most probability mass lands on the digit tokens.
  - HARD (the parser): anything that is not a valid class index is discarded.

Run:
    uv run python tests/demo_extraction.py
    uv run python tests/demo_extraction.py --teacher gpt-4.1-mini
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

    # 2. The CLEAN k-class vector the parser produces from that mess.
    logits, floored = parse_logprobs_to_logits(response, verbalizers)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    print(f"\n  CLEAN {num_classes}-class output (after the hard constraint / parser):")
    for i, name in enumerate(label_names):
        floor_note = " (floor: never appeared in top-k)" if floored[i] else ""
        print(f"    class {i} {name:<10} logit={logits[i]:>8.3f}  prob={probs[i]:>7.2%}{floor_note}")

    pred = int(logits.argmax())
    mark = "correct" if pred == true_label else "WRONG"
    print(f"\n  Prediction : class {pred} = {label_names[pred]}  ({mark})")
    print("-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo logit extraction on AG News samples.")
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

    ag = load_dataset_by_name("ag_news")
    client = get_client(config)
    verbalizers = make_verbalizers(ag.num_classes)
    fewshot_pool = build_fewshot_pool(ag)

    print(f"Dataset: AG News  |  classes: {ag.label_names}")
    print(f"Teacher: {config.name} (model={config.model})\n")
    print("=" * 70)

    for i in range(N_SAMPLES):
        sample = ag.train[i]
        text = sample[ag.text_column]
        true_label = sample[ag.label_column]

        messages = build_prompt(
            text,
            ag.label_names,
            verbalizers,
            fewshot_examples_for_index(ag, verbalizers, fewshot_pool, i),
        )
        response = complete(
            client,
            config,
            messages,
            guided_choices=verbalizers if config.constrain_to_verbalizers else None,
        )

        print(f"SAMPLE {i}")
        demo_sample(text, true_label, ag.label_names, response)
        print()


if __name__ == "__main__":
    main()
