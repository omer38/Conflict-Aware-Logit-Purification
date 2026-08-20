#!/usr/bin/env python
"""Constrain OpenAI generation toward class verbalizers and read class logits.

OpenAI returns logprobs only for the top-k tokens in the full vocabulary
(default k=20), not for every class letter. Classes missing from that list
get an imputed floor (same logic as parse_logprobs_to_logits).

We apply logit_bias (+100) on class token IDs via tiktoken. That strongly
favours class letters at decode time but does not make the API return all
class logprobs automatically — you still only see tokens that land in top-k.

Usage:
    uv run python scripts/constrained_class_probs.py --dataset ag_news
    uv run python scripts/constrained_class_probs.py --dataset ag_news --teacher gpt-4.1-mini --samples 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletion

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.teachers.extraction import (
    LOGPROB_FLOOR,
    build_prompt,
    build_fewshot_pool,
    fewshot_examples_for_index,
    make_verbalizers,
    parse_logprobs_to_logits,
)
from src.teachers.llm import TEACHER_PRESETS, get_client, complete

_OPENAI_TEACHERS = ("gpt-4.1-mini", "gpt-4o")
_CLASS_BIAS = 100


def _openai_teacher_choices() -> list[str]:
    return [name for name in _OPENAI_TEACHERS if name in TEACHER_PRESETS]


def resolve_class_token_ids(model: str, verbalizers: list[str]) -> dict[str, int]:
    """Map each verbalizer string to its single tiktoken ID for *model*."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")

    token_ids: dict[str, int] = {}
    for verbalizer in verbalizers:
        ids = enc.encode(verbalizer)
        if len(ids) != 1:
            raise ValueError(
                f"Verbalizer {verbalizer!r} encodes to {ids!r}, expected a single token"
            )
        token_ids[verbalizer] = ids[0]
    return token_ids


def build_class_logit_bias(token_ids: dict[str, int], bias: float = _CLASS_BIAS) -> dict[str, int]:
    """Return OpenAI logit_bias dict boosting class token IDs."""
    value = int(bias)
    if not -100 <= value <= 100:
        raise ValueError("logit_bias values must be between -100 and 100")
    return {str(token_id): value for token_id in token_ids.values()}


def complete_constrained(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    logit_bias: dict[str, int],
    top_logprobs: int,
) -> ChatCompletion:
    """Single-token completion with class-token logit_bias and logprobs."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=top_logprobs,
        logit_bias=logit_bias,
    )


def class_logits_from_response(
    response: ChatCompletion,
    verbalizers: list[str],
    token_ids: dict[str, int],
    enc: tiktoken.Encoding,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build a C-dimensional logit vector from top_logprobs using token IDs.

    Matches entries by exact tiktoken ID (not stripped strings). Classes not
    present in top-k receive the same adaptive floor used in extraction.py.
    """
    id_to_class = {token_ids[v]: i for i, v in enumerate(verbalizers)}
    allowed_ids = set(id_to_class)

    choice = response.choices[0]
    content_logprobs = choice.logprobs.content if choice.logprobs else []
    if not content_logprobs:
        raise RuntimeError("No logprobs in response")

    top = content_logprobs[0].top_logprobs
    raw_tokens = [entry.token for entry in top]
    floor = min(min(entry.logprob for entry in top) - 1.0, LOGPROB_FLOOR)

    logits = np.full(len(verbalizers), floor, dtype=np.float32)
    observed = np.zeros(len(verbalizers), dtype=bool)

    for entry in top:
        ids = enc.encode(entry.token)
        if len(ids) != 1 or ids[0] not in allowed_ids:
            continue
        idx = id_to_class[ids[0]]
        if not observed[idx] or entry.logprob > logits[idx]:
            logits[idx] = entry.logprob
            observed[idx] = True

    return logits, observed, raw_tokens


def print_logits_table(
    label_names: list[str],
    verbalizers: list[str],
    logits: np.ndarray,
    observed: np.ndarray,
) -> None:
    """Print per-class logit values and whether each came from the API."""
    print(f"    {'class':<6} {'name':<12} {'logit':>10}  source")
    for i, name in enumerate(label_names):
        source = "API top_logprobs" if observed[i] else f"imputed floor (not in top-k)"
        print(f"    {verbalizers[i]:<6} {name:<12} {logits[i]:>10.4f}  {source}")


def print_sample_report(
    sample_idx: int,
    text: str,
    true_label: int,
    label_names: list[str],
    verbalizers: list[str],
    token_ids: dict[str, int],
    top_logprobs: int,
    unconstrained: ChatCompletion,
    constrained: ChatCompletion,
    enc: tiktoken.Encoding,
) -> None:
    """Print side-by-side unconstrained vs constrained class logits."""
    u_logits, u_observed = _logits_with_observed(unconstrained, verbalizers)
    c_logits, c_observed, c_top = class_logits_from_response(
        constrained, verbalizers, token_ids, enc
    )

    u_choice = unconstrained.choices[0]
    c_choice = constrained.choices[0]
    u_top = u_choice.logprobs.content[0].top_logprobs if u_choice.logprobs else []
    u_raw = [e.token for e in u_top]

    print(f"\n{'=' * 72}")
    print(f"SAMPLE {sample_idx}")
    print(f'  Text       : "{text[:90]}{"..." if len(text) > 90 else ""}"')
    print(f"  True class : {true_label} = {label_names[true_label]}")
    print(f"  Token IDs  : {{{', '.join(f'{v}:{token_ids[v]}' for v in verbalizers)}}}")
    print(
        f"\n  Why some classes have no API logit: OpenAI only returns the top-{top_logprobs} "
        "tokens from the full vocabulary. Other class letters may rank below that cutoff "
        "and never appear in the response."
    )

    print(f"\n  UNCONSTRAINED (full-vocab generation, then parser keeps class letters)")
    print(f'    Generated : {u_choice.message.content!r}')
    print(f"    Raw top-{len(u_raw)} first tokens: {u_raw}")
    print_logits_table(label_names, verbalizers, u_logits, u_observed)
    u_pred = int(u_logits.argmax())
    print(f"    Argmax    : {u_pred} = {label_names[u_pred]}")

    print(f"\n  CONSTRAINED (logit_bias +100 on class token IDs, max_tokens=1)")
    print(f'    Generated : {c_choice.message.content!r}')
    print(f"    Raw top-{len(c_top)} first tokens: {c_top}")
    print_logits_table(label_names, verbalizers, c_logits, c_observed)
    c_pred = int(c_logits.argmax())
    print(f"    Argmax    : {c_pred} = {label_names[c_pred]}")


def _logits_with_observed(
    response: ChatCompletion,
    verbalizers: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Wrap parse_logprobs_to_logits and invert the floored mask."""
    logits, floored = parse_logprobs_to_logits(response, verbalizers)
    return logits, ~floored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI class logits via logit_bias on verbalizer token IDs."
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
        choices=_openai_teacher_choices(),
        help="OpenAI teacher preset (must use api.openai.com).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of training samples to demo.",
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=20,
        help="How many tokens OpenAI returns per step (max 20 on most models).",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=_CLASS_BIAS,
        help="logit_bias value applied to each class token ID (default 100).",
    )
    args = parser.parse_args()

    config = TEACHER_PRESETS[args.teacher]
    if not os.environ.get(config.api_key_env):
        print(f"{config.api_key_env} is not set. Add it to .env or the environment.")
        sys.exit(1)

    dataset = load_dataset_by_name(args.dataset)
    verbalizers = make_verbalizers(dataset.num_classes)
    try:
        enc = tiktoken.encoding_for_model(config.model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    token_ids = resolve_class_token_ids(config.model, verbalizers)
    logit_bias = build_class_logit_bias(token_ids, bias=args.bias)
    client = get_client(config)
    fewshot_pool = build_fewshot_pool(dataset)

    print(f"Teacher  : {config.name} (model={config.model})")
    print(f"Dataset  : {dataset.name} ({dataset.num_classes} classes)")
    print(f"Classes  : {list(zip(verbalizers, dataset.label_names))}")
    print(f"logit_bias: +{int(args.bias)} on token IDs {list(token_ids.values())}")
    print(f"top_logprobs: {args.top_logprobs}")
    print(
        "Note: OpenAI allows at most 300 logit_bias entries, so we boost class "
        "tokens rather than suppressing the full vocabulary."
    )

    n = min(args.samples, len(dataset.train))
    for i in range(n):
        sample = dataset.train[i]
        text = sample[dataset.text_column]
        true_label = int(sample[dataset.label_column])
        examples = fewshot_examples_for_index(dataset, verbalizers, fewshot_pool, i)
        messages = build_prompt(text, dataset.label_names, verbalizers, examples)

        unconstrained = complete(client, config, messages)
        constrained = complete_constrained(
            client,
            config.model,
            messages,
            logit_bias,
            top_logprobs=args.top_logprobs,
        )

        print_sample_report(
            i,
            text,
            true_label,
            dataset.label_names,
            verbalizers,
            token_ids,
            args.top_logprobs,
            unconstrained,
            constrained,
            enc,
        )

    print(f"\n{'=' * 72}")
    print("Done.")


if __name__ == "__main__":
    main()
