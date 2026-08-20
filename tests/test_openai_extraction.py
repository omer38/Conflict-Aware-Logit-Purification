"""Live API test for the OpenAI logit extraction pipeline.

Sends a handful of AG News samples to gpt-4.1-mini and checks that the
returned logits are well-formed. Requires OPENAI_API_KEY to be set
(reads from .env if present).

Run:
    uv run python tests/test_openai_extraction.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loading import load_dataset_by_name
from src.teachers.llm import TEACHER_PRESETS, get_client, complete
from src.teachers.extraction import (
    build_prompt,
    build_fewshot_pool,
    extract_logits_for_dataset,
    fewshot_examples_for_index,
    load_cached_logits,
    make_verbalizers,
    parse_logprobs_to_logits,
)


def load_env() -> None:
    """Load .env file into os.environ if it exists."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def test_single_sample() -> None:
    """Send one AG News sample to gpt-4.1-mini and verify the logit vector."""
    print("--- test_single_sample ---")

    config = TEACHER_PRESETS["gpt-4.1-mini"]
    ag = load_dataset_by_name("ag_news")
    sample = ag.train[0]
    text = sample[ag.text_column]
    true_label = sample[ag.label_column]

    print(f"  Text:  \"{text[:80]}...\"")
    print(f"  Label: {true_label} ({ag.label_names[true_label]})")

    verbalizers = make_verbalizers(ag.num_classes)
    pool = build_fewshot_pool(ag)
    examples = fewshot_examples_for_index(ag, verbalizers, pool, 0)
    messages = build_prompt(text, ag.label_names, verbalizers, examples)
    client = get_client(config)
    response = complete(client, config, messages)

    # Check raw response structure.
    choice = response.choices[0]
    reply_text = choice.message.content.strip()
    print(f"  Reply: \"{reply_text}\"")

    assert choice.logprobs is not None, "logprobs missing from response"
    assert len(choice.logprobs.content) > 0, "logprobs.content is empty"

    first_token = choice.logprobs.content[0]
    print(f"  Top logprobs ({len(first_token.top_logprobs)} entries):")
    for entry in first_token.top_logprobs[:6]:
        print(f"    token={entry.token!r:>5}  logprob={entry.logprob:.4f}")

    # Parse into logit vector.
    logits, floored = parse_logprobs_to_logits(response, verbalizers)
    assert logits.shape == (4,), f"expected shape (4,), got {logits.shape}"

    # For 4 classes with top_logprobs=20, all classes should be observed.
    floor_count = floored.sum()
    print(f"  Logits: {logits}")
    print(f"  Floor values: {floor_count}/4")
    assert floor_count == 0, f"AG News has only 4 classes but {floor_count} got floor value"

    # Softmax sanity.
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    assert abs(probs.sum() - 1.0) < 1e-5, f"probs sum to {probs.sum()}"
    print(f"  Probs:  {probs}")
    print(f"  Argmax: {logits.argmax()} (true: {true_label})")
    print("  PASSED\n")


def test_batch_with_cache() -> None:
    """Extract 5 samples, save to cache, reload, and verify resume skips API calls."""
    print("--- test_batch_with_cache ---")

    config = TEACHER_PRESETS["gpt-4.1-mini"]
    ag = load_dataset_by_name("ag_news")

    with tempfile.TemporaryDirectory() as tmpdir:
        # First run: extract 5 samples.
        logits = extract_logits_for_dataset(config, ag, cache_dir=tmpdir, limit=5)
        assert logits.shape == (len(ag.train), ag.num_classes)

        cached = load_cached_logits(config.name, ag.name, tmpdir)
        assert cached is not None
        logits_loaded, mask, floored = cached
        assert mask[:5].all(), "first 5 should be marked extracted"
        assert not mask[5:].any(), "rest should not be extracted"

        # Verify logit values for extracted rows.
        for i in range(5):
            row = logits_loaded[i]
            floor_count = floored[i].sum()
            assert floor_count == 0, f"sample {i}: {floor_count}/4 floor values"

            probs = np.exp(row - row.max())
            probs = probs / probs.sum()
            assert abs(probs.sum() - 1.0) < 1e-5

        print(f"  Extracted 5 samples, all valid.")

        # Second run: should skip all 5, making zero API calls.
        logits2 = extract_logits_for_dataset(config, ag, cache_dir=tmpdir, limit=5)
        assert np.array_equal(logits[:5], logits2[:5]), "cached logits changed on re-run"
        print(f"  Re-run skipped correctly (cache hit).")

    print("  PASSED\n")


def test_accuracy_sanity() -> None:
    """Check that gpt-4.1-mini gets reasonable accuracy on 10 AG News samples."""
    print("--- test_accuracy_sanity ---")

    config = TEACHER_PRESETS["gpt-4.1-mini"]
    ag = load_dataset_by_name("ag_news")

    with tempfile.TemporaryDirectory() as tmpdir:
        logits = extract_logits_for_dataset(config, ag, cache_dir=tmpdir, limit=10)
        predictions = logits[:10].argmax(axis=1)
        labels = np.array(ag.train[:10][ag.label_column])
        accuracy = (predictions == labels).mean()

        print(f"  Predictions: {predictions.tolist()}")
        print(f"  Labels:      {labels.tolist()}")
        print(f"  Accuracy:    {accuracy:.0%} (10 samples)")

        # gpt-4.1-mini should get at least 50% on 10 easy AG News samples.
        assert accuracy >= 0.5, f"accuracy {accuracy:.0%} is suspiciously low"

    print("  PASSED\n")


def main() -> None:
    load_env()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Skipping live API tests.")
        sys.exit(0)

    test_single_sample()
    test_batch_with_cache()
    test_accuracy_sanity()

    print("=== All tests passed. ===")


if __name__ == "__main__":
    main()
