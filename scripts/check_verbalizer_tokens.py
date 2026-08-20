#!/usr/bin/env python
"""Check whether class verbalizers are single first-output tokens.

This uses the configured teacher API directly instead of a local tokenizer, so it
checks the exact provider/model path used during extraction. It validates the same
single-letter verbalizers (A, B, ...) that the extraction pipeline uses.

Usage:
    uv run python scripts/check_verbalizer_tokens.py --dataset amazon --teacher all
    uv run python scripts/check_verbalizer_tokens.py --dataset yahoo --teacher gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.teachers.extraction import make_verbalizers
from src.teachers.llm import TEACHER_PRESETS, TeacherConfig, complete, get_client, has_api_credentials


def load_env() -> None:
    """Load .env into os.environ if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_probe_prompt(label: str) -> list[dict[str, str]]:
    """Ask the model to emit one known verbalizer."""
    return [
        {
            "role": "system",
            "content": "You copy classification labels exactly. Reply with only the requested label.",
        },
        {
            "role": "user",
            "content": f"Requested label: {label}\nReply with exactly that label and nothing else.",
        },
    ]


def check_teacher(config: TeacherConfig, labels: list[str]) -> bool:
    """Return True if every label appears as the first generated token."""
    if not has_api_credentials(config):
        print(f"\n{config.name}: skipped ({config.api_key_env} is not set)")
        return True

    print(f"\n{config.name} ({config.model})")
    print("-" * 70)

    client = get_client(config)
    failures: list[tuple[str, str, list[str]]] = []
    errors: list[tuple[str, object]] = []

    for label in labels:
        response = complete(client, config, build_probe_prompt(label))

        # OpenRouter returns choices=None with an error payload when the routed
        # backend fails (rate limit, upstream error). Treat as inconclusive,
        # not as a verbalizer failure, and keep checking the rest.
        if not response.choices:
            error = getattr(response, "error", None) or (response.model_extra or {}).get("error")
            print(f"  {label!r:<4} ERR  no choices returned (provider error: {error})")
            errors.append((label, error))
            continue

        choice = response.choices[0]
        generated = (choice.message.content or "").strip()
        content_logprobs = choice.logprobs.content if choice.logprobs else []
        tokens = [entry.token for entry in content_logprobs]
        first_token = tokens[0].strip() if tokens else ""

        ok = generated == label and first_token == label
        status = "OK" if ok else "FAIL"
        print(
            f"  {label!r:<4} {status:<4} generated={generated!r:<6} "
            f"first_token={first_token!r:<6} tokens={tokens!r}"
        )

        if not ok:
            failures.append((label, generated, tokens))

    if failures:
        print(f"\n  FAILED: {len(failures)}/{len(labels)} labels are unsafe for first-token parsing.")
        for label, generated, tokens in failures:
            print(f"    label={label!r} generated={generated!r} tokens={tokens!r}")
        return False

    if errors:
        print(
            f"\n  INCONCLUSIVE: {len(errors)}/{len(labels)} labels errored (provider issue); "
            "re-run to verify them."
        )
        for label, error in errors:
            print(f"    label={label!r} error={error}")
        return False

    print(f"\n  PASSED: all {len(labels)} labels are single first-output tokens.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether numeric class labels are safe single-token verbalizers."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["amazon", "yahoo", "ag_news"],
        help="Dataset whose class count determines the labels to check.",
    )
    parser.add_argument(
        "--teacher",
        default="all",
        choices=["all", *TEACHER_PRESETS.keys()],
        help="Teacher preset to check, or all configured teachers.",
    )
    args = parser.parse_args()

    load_env()

    dataset = load_dataset_by_name(args.dataset)
    labels = make_verbalizers(dataset.num_classes)
    print(f"Dataset: {dataset.name} ({dataset.num_classes} classes)")
    print(f"Labels:  {labels}")

    if args.teacher == "all":
        configs = list(TEACHER_PRESETS.values())
    else:
        configs = [TEACHER_PRESETS[args.teacher]]

    all_ok = True
    for config in configs:
        all_ok = check_teacher(config, labels) and all_ok

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
