#!/usr/bin/env python
"""Check GPT-4.1 mini classification accuracy on Amazon review full.

Usage:
    uv run python scripts/check_amazon_gpt41mini_accuracy.py
    uv run python scripts/check_amazon_gpt41mini_accuracy.py --seed 7

Set OPENAI_API_KEY in the environment or in a local .env file before running.
Accuracy was 62.5%

Model:   gpt-4.1-mini
Dataset: vgaraujov/amazon_review_full [test]
Samples: 200
200/200  accuracy=62.50%

Correct:  125/200
Accuracy: 62.50%
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-4.1-mini"
DATASET = "vgaraujov/amazon_review_full"
SAMPLE_COUNT = 200
LABEL_NAMES = [
    "1 star",
    "2 stars",
    "3 stars",
    "4 stars",
    "5 stars",
]


def build_messages(title: str, content: str) -> list[dict[str, str]]:
    labels = "\n".join(f"{idx}: {name}" for idx, name in enumerate(LABEL_NAMES))
    return [
        {
            "role": "system",
            "content": (
                "You are an Amazon review sentiment classifier. "
                "Reply with only one integer label ID and no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Classify this Amazon review into the correct star-rating class.\n\n"
                f"Labels:\n{labels}\n\n"
                f"Title:\n{title}\n\n"
                f"Review:\n{content}\n\n"
                "Answer with only one integer from 0 to 4."
            ),
        },
    ]


def parse_label(output: str) -> int | None:
    match = re.search(r"\b[0-4]\b", output.strip())
    if match is None:
        return None
    return int(match.group(0))


def predict_label(client: OpenAI, title: str, content: str) -> tuple[int | None, str]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=build_messages(title, content),
        temperature=0,
        max_tokens=4,
    )
    output = response.choices[0].message.content or ""
    return parse_label(output), output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate gpt-4.1-mini accuracy on 200 random Amazon review full samples."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42).",
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Add it to .env or the environment.")
        sys.exit(1)

    dataset = load_dataset(DATASET, split=args.split)
    dataset = dataset.shuffle(seed=args.seed).select(range(SAMPLE_COUNT))

    client = OpenAI()
    correct = 0
    failures: list[tuple[int, int, int | None, str]] = []

    print(f"Model:   {MODEL}")
    print(f"Dataset: {DATASET} [{args.split}]")
    print(f"Samples: {len(dataset)}")

    for idx, row in enumerate(dataset, start=1):
        expected = int(row["label"])
        predicted, raw_output = predict_label(client, row["title"], row["content"])
        is_correct = predicted == expected
        correct += int(is_correct)

        if not is_correct and len(failures) < 10:
            failures.append((idx, expected, predicted, raw_output.strip()))

        accuracy = correct / idx
        print(
            f"\r{idx}/{len(dataset)}  accuracy={accuracy:.2%}",
            end="",
            flush=True,
        )

    accuracy = correct / len(dataset) if len(dataset) else 0.0
    print("\n")
    print(f"Correct:  {correct}/{len(dataset)}")
    print(f"Accuracy: {accuracy:.2%}")

    if failures:
        print("\nFirst errors:")
        for idx, expected, predicted, raw_output in failures:
            expected_name = LABEL_NAMES[expected]
            predicted_name = LABEL_NAMES[predicted] if predicted is not None else "unparsed"
            print(
                f"  #{idx}: expected={expected} ({expected_name}), "
                f"predicted={predicted} ({predicted_name}), raw={raw_output!r}"
            )


if __name__ == "__main__":
    main()
