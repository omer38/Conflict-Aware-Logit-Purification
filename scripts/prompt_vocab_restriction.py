#!/usr/bin/env python
"""Observe how a strict system prompt reshapes the first-token top-20 logprobs.

Lever 1 from the analysis: tighten the prompt so the model stops spending its
probability mass on formatting variants of the winning letter (' A', '\\tA',
'"A', '[A', homoglyphs, ...) and instead leaves those top-20 slots for the
other class letters. No logit_bias is used here -- this isolates the pure
effect of the prompt on the natural distribution.

For each sample it prints, for both the DEFAULT and the STRICT prompt:
  - the raw top-20 first tokens the API returned
  - the per-class logits (observed vs imputed floor)
so you can see whether more class letters become observed.

Usage:
    uv run python scripts/prompt_vocab_restriction.py --dataset ag_news
    uv run python scripts/prompt_vocab_restriction.py --dataset ag_news --teacher gemma4-31b-mlx
    uv run python scripts/prompt_vocab_restriction.py --dataset amazon --samples 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.teachers.extraction import (
    build_prompt,
    build_fewshot_pool,
    fewshot_examples_for_index,
    make_verbalizers,
    parse_logprobs_to_logits,
)
from src.teachers.llm import TEACHER_PRESETS, get_client, complete, _OLLAMA_BASE_URL


def _teacher_has_credentials(config) -> bool:
    if config.base_url == _OLLAMA_BASE_URL:
        return True
    return bool(os.environ.get(config.api_key_env))


def build_default_prompt(
    text: str,
    label_names: list[str],
    verbalizers: list[str],
) -> list[dict[str, str]]:
    """Original soft-constraint prompt (kept here for A/B comparison only)."""
    categories = "\n".join(f"{v}: {name}" for v, name in zip(verbalizers, label_names))
    return [
        {
            "role": "system",
            "content": (
                "You are a text classifier. "
                "Reply with ONLY the category letter, nothing else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Classify the following text into one of the categories below.\n\n"
                f"Categories:\n{categories}\n\n"
                f'Text: "{text}"'
            ),
        },
    ]


def _strict_rules(letters: str) -> str:
    """Shared format-pinning system rules used by strict and few-shot prompts."""
    return (
        "You are a strict single-token text classifier.\n"
        f"The ONLY valid outputs are these single characters: {letters}.\n"
        "Output exactly one character, in plain ASCII uppercase.\n"
        "No leading or trailing spaces. No quotes, punctuation, markdown, "
        "or explanation.\n"
        f"Never output anything other than one of: {letters}."
    )


def build_strict_prompt(
    text: str,
    label_names: list[str],
    verbalizers: list[str],
) -> list[dict[str, str]]:
    """Strict rules only, no few-shot (A/B comparison)."""
    letters = ", ".join(verbalizers)
    categories = "\n".join(f"{v}: {name}" for v, name in zip(verbalizers, label_names))
    return [
        {"role": "system", "content": _strict_rules(letters)},
        {
            "role": "user",
            "content": (
                f"Classify the following text into one of the categories below.\n\n"
                f"Categories:\n{categories}\n\n"
                f'Text: "{text}"'
            ),
        },
    ]


def print_style_block(
    title: str,
    response,
    label_names: list[str],
    verbalizers: list[str],
) -> int:
    """Print raw top-k tokens and per-class logits. Returns observed class count."""
    choice = response.choices[0]
    top = choice.logprobs.content[0].top_logprobs if choice.logprobs else []
    raw = [e.token for e in top]
    logits, floored = parse_logprobs_to_logits(response, verbalizers)
    observed = ~floored

    print(f"\n  {title}")
    print(f'    Generated : {choice.message.content!r}')
    print(f"    Raw top-{len(raw)} first tokens: {raw}")
    print(f"    {'class':<6} {'name':<12} {'logit':>10}  source")
    for i, name in enumerate(label_names):
        source = "API top_logprobs" if observed[i] else "imputed floor (not in top-k)"
        print(f"    {verbalizers[i]:<6} {name:<12} {logits[i]:>10.4f}  {source}")
    pred = int(logits.argmax())
    print(f"    Observed  : {int(observed.sum())}/{len(verbalizers)} classes")
    print(f"    Argmax    : {pred} = {label_names[pred]}")
    return int(observed.sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare default vs strict prompt on first-token top-20 logprobs."
    )
    parser.add_argument(
        "--dataset",
        default="ag_news",
        choices=["amazon", "yahoo", "ag_news"],
        help="Dataset for sample texts and class count.",
    )
    parser.add_argument(
        "--teacher",
        default="gpt-4.1-mini",
        choices=list(TEACHER_PRESETS.keys()),
        help="Teacher preset (OpenAI, OpenRouter, Ollama, etc.).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of training samples to demo.",
    )
    args = parser.parse_args()

    config = TEACHER_PRESETS[args.teacher]
    if not _teacher_has_credentials(config):
        print(f"{config.api_key_env} is not set. Add it to .env or the environment.")
        sys.exit(1)

    dataset = load_dataset_by_name(args.dataset)
    verbalizers = make_verbalizers(dataset.num_classes)
    client = get_client(config)
    fewshot_pool = build_fewshot_pool(dataset)

    n = min(args.samples, len(dataset.train))

    print(f"Teacher  : {config.name} (model={config.model})")
    print(f"Dataset  : {dataset.name} ({dataset.num_classes} classes)")
    print(f"Classes  : {list(zip(verbalizers, dataset.label_names))}")
    print(f"top_logprobs: {config.max_top_logprobs}  (no logit_bias; prompt effect only)")
    print(f"Production prompt: strict + few-shot via src.teachers.extraction.build_prompt")

    default_observed = 0
    strict_observed = 0
    fewshot_observed = 0
    total = n * dataset.num_classes

    for i in range(n):
        sample = dataset.train[i]
        text = sample[dataset.text_column]
        true_label = int(sample[dataset.label_column])

        default_msgs = build_default_prompt(text, dataset.label_names, verbalizers)
        strict_msgs = build_strict_prompt(text, dataset.label_names, verbalizers)
        prod_examples = fewshot_examples_for_index(dataset, verbalizers, fewshot_pool, i)
        prod_msgs = build_prompt(text, dataset.label_names, verbalizers, prod_examples)

        default_resp = complete(client, config, default_msgs)
        strict_resp = complete(client, config, strict_msgs)
        prod_resp = complete(client, config, prod_msgs)

        print(f"\n{'=' * 72}")
        print(f"SAMPLE {i}")
        print(f'  Text       : "{text[:90]}{"..." if len(text) > 90 else ""}"')
        print(f"  True class : {true_label} = {dataset.label_names[true_label]}")

        default_observed += print_style_block(
            "DEFAULT PROMPT", default_resp, dataset.label_names, verbalizers
        )
        strict_observed += print_style_block(
            "STRICT PROMPT", strict_resp, dataset.label_names, verbalizers
        )
        fewshot_observed += print_style_block(
            f"PRODUCTION (strict + few-shot, {len(prod_examples)} shots)",
            prod_resp,
            dataset.label_names,
            verbalizers,
        )

    print(f"\n{'=' * 72}")
    print("SUMMARY (observed class logits = not floored)")
    print(f"  Default prompt    : {default_observed}/{total} ({default_observed / total:.1%})")
    print(f"  Strict prompt     : {strict_observed}/{total} ({strict_observed / total:.1%})")
    print(f"  Production prompt : {fewshot_observed}/{total} ({fewshot_observed / total:.1%})")
    print("Done.")


if __name__ == "__main__":
    main()
