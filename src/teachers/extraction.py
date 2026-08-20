"""Classification-specific logit extraction from teacher LLMs.

Builds prompts, parses logprobs into C-dimensional logit vectors,
and manages an NPZ cache for resumable extraction.
"""

from __future__ import annotations

import csv
import json
import string
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from openai.types.chat import ChatCompletion

from .llm import TeacherConfig, complete, get_client, is_local_vllm, wait_until_ready
from ..data.loading import TextClassificationDataset
from ..data.subsampling import PROMPT_DEMOS_PER_CLASS, prompt_demo_indices

LOGPROB_FLOOR = -30.0

# Single uppercase letters are single tokens on every teacher tokenizer we use
# (verified via scripts/check_verbalizer_tokens.py), unlike multi-digit numbers
# such as "10", which Gemma/Qwen split into ["1", "0"] and the first-token
# parser cannot reconstruct.
MAX_VERBALIZERS = len(string.ascii_uppercase)


def make_verbalizers(num_classes: int) -> list[str]:
    """Return one single-token verbalizer per class (``["A", "B", ...]``).

    Letters are used instead of class numbers so each label maps to exactly one
    first output token on every teacher. Raises if there are more classes than
    available single letters; larger label sets (e.g. full Banking77) need a
    different multi-token-safe scheme.
    """
    if num_classes > MAX_VERBALIZERS:
        raise ValueError(
            f"{num_classes} classes exceeds the {MAX_VERBALIZERS} single-letter "
            "verbalizers (A-Z). Add a multi-token-safe scheme for larger label sets."
        )
    return list(string.ascii_uppercase[:num_classes])


def _strict_rules(letters: str) -> str:
    """Format-pinning system rules for strict / few-shot classification prompts."""
    return (
        "You are a strict single-token text classifier.\n"
        f"The ONLY valid outputs are these single characters: {letters}.\n"
        "Output exactly one character, in plain ASCII uppercase.\n"
        "No leading or trailing spaces. No quotes, punctuation, markdown, "
        "or explanation.\n"
        f"Never output anything other than one of: {letters}."
    )


def build_fewshot_pool(
    dataset: TextClassificationDataset,
    per_class: int = PROMPT_DEMOS_PER_CLASS,
) -> dict[int, list[int]]:
    """Map classes to the fixed demonstrations counted in every declared D_L."""
    return prompt_demo_indices(dataset, per_class)


def fewshot_examples_for_index(
    dataset: TextClassificationDataset,
    verbalizers: list[str],
    pool: dict[int, list[int]],
    idx: int,
) -> list[tuple[str, str]]:
    """Return one exemplar per class, preferring indices other than *idx*."""
    train = dataset.train
    examples: list[tuple[str, str]] = []
    for class_id in range(dataset.num_classes):
        pick = next((i for i in pool[class_id] if i != idx), pool[class_id][0])
        text = train[pick][dataset.text_column]
        examples.append((text, verbalizers[class_id]))
    return examples


def build_prompt(
    text: str,
    label_names: list[str],
    verbalizers: list[str],
    examples: list[tuple[str, str]],
) -> list[dict[str, str]]:
    """Build strict + few-shot classification messages for logit extraction.

    Exemplar assistant turns end in a single bare class letter so the model
    learns the exact output token form and more class letters appear in top-k.
    """
    letters = ", ".join(verbalizers)
    categories = "\n".join(f"{v}: {name}" for v, name in zip(verbalizers, label_names))
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": f"{_strict_rules(letters)}\n\nCategories:\n{categories}",
        }
    ]
    for ex_text, ex_letter in examples:
        messages.append({"role": "user", "content": f'Text: "{ex_text}"'})
        messages.append({"role": "assistant", "content": ex_letter})
    messages.append({"role": "user", "content": f'Text: "{text}"'})
    return messages


def parse_logprobs_to_logits(
    response: ChatCompletion,
    verbalizers: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a C-dimensional logit vector from the API response.

    Reads the first output token's top_logprobs and maps each verbalizer
    letter to its class index. Classes not observed in top-k get an adaptive
    floor: min(lowest observed logprob - 1, LOGPROB_FLOOR). This guarantees
    imputed values always rank below every observed token (real logprobs can
    fall below a constant -30, which would otherwise rank above them).

    Returns (logits, floored) where floored is a boolean mask marking the
    classes whose values were imputed rather than observed.
    """
    token_to_class = {v: i for i, v in enumerate(verbalizers)}

    token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    floor = min(min(e.logprob for e in token_logprobs) - 1.0, LOGPROB_FLOOR)
    logits = np.full(len(verbalizers), floor, dtype=np.float32)
    floored = np.ones(len(verbalizers), dtype=bool)

    for entry in token_logprobs:
        idx = token_to_class.get(entry.token.strip())
        if idx is not None and floored[idx]:
            logits[idx] = entry.logprob
            floored[idx] = False

    return logits, floored


def _cache_path(teacher_name: str, dataset_name: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"{teacher_name}_{dataset_name}.npz"


def load_cached_logits(
    teacher_name: str,
    dataset_name: str,
    cache_dir: str = "cache/logits",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load cached logits from disk.

    Returns (logits, extracted_mask, floored) or None if no cache exists.
    """
    path = _cache_path(teacher_name, dataset_name, cache_dir)
    if not path.exists():
        return None
    data = np.load(path)
    logits = data["logits"]
    if "floored" in data.files:
        floored = data["floored"]
    else:
        # Legacy caches used a constant floor, so equality identifies it.
        floored = logits == LOGPROB_FLOOR
    return logits, data["extracted_mask"], floored


def _save_cache(
    logits: np.ndarray,
    extracted_mask: np.ndarray,
    floored: np.ndarray,
    teacher_name: str,
    dataset_name: str,
    cache_dir: str,
) -> None:
    path = _cache_path(teacher_name, dataset_name, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, logits=logits, extracted_mask=extracted_mask, floored=floored)


def extract_logits_for_dataset(
    config: TeacherConfig,
    dataset: TextClassificationDataset,
    cache_dir: str = "cache/logits",
    limit: int | None = None,
    save_every: int = 50,
) -> np.ndarray:
    """Extract logits for every training sample, with resume support.

    Parameters
    ----------
    config:
        Teacher model configuration.
    dataset:
        The dataset whose training split will be queried.
    cache_dir:
        Directory for NPZ cache files.
    limit:
        If set, only extract the first *limit* samples (for testing).
    save_every:
        Save cache to disk every this many new extractions.

    Returns
    -------
    logits array of shape (N, C), dtype float32.
    """
    n_samples = len(dataset.train)
    n_classes = dataset.num_classes

    cached = load_cached_logits(config.name, dataset.name, cache_dir)
    if cached is not None:
        logits, extracted_mask, floored = cached
    else:
        logits = np.full((n_samples, n_classes), LOGPROB_FLOOR, dtype=np.float32)
        extracted_mask = np.zeros(n_samples, dtype=bool)
        floored = np.zeros((n_samples, n_classes), dtype=bool)

    indices = range(min(limit, n_samples) if limit else n_samples)
    todo = [i for i in indices if not extracted_mask[i]]

    if not todo:
        print(f"All {sum(extracted_mask)} samples already cached. Skipping.")
        return logits

    print(f"Extracting {len(todo)} samples ({sum(extracted_mask)} already cached)...")
    print("Prompt: strict + few-shot (one exemplar per class; re-extract if cache predates this change)")
    client = get_client(config)
    if is_local_vllm(config):
        wait_until_ready(client, config)
    verbalizers = make_verbalizers(n_classes)
    fewshot_pool = build_fewshot_pool(dataset)
    new_count = 0

    for i, idx in enumerate(todo):
        text = dataset.train[idx][dataset.text_column]
        examples = fewshot_examples_for_index(dataset, verbalizers, fewshot_pool, idx)
        messages = build_prompt(text, dataset.label_names, verbalizers, examples)

        response = complete(
            client,
            config,
            messages,
            guided_choices=verbalizers if config.constrain_to_verbalizers else None,
        )
        logits[idx], floored[idx] = parse_logprobs_to_logits(response, verbalizers)
        extracted_mask[idx] = True
        new_count += 1

        if new_count % save_every == 0:
            _save_cache(logits, extracted_mask, floored, config.name, dataset.name, cache_dir)
            print(f"  [{new_count}/{len(todo)}] saved checkpoint")

    _save_cache(logits, extracted_mask, floored, config.name, dataset.name, cache_dir)
    print(f"Done. Extracted {new_count} new samples, {sum(extracted_mask)} total cached.")
    return logits


def default_report_path(
    teacher_name: str,
    dataset_name: str,
    report_dir: str = "outputs/reports",
) -> Path:
    """Default CSV path for a teacher/dataset extraction report."""
    return Path(report_dir) / f"{teacher_name}_{dataset_name}.csv"


def write_extraction_report(
    logits: np.ndarray,
    dataset: TextClassificationDataset,
    teacher_name: str,
    report_path: str | Path,
    *,
    extracted_mask: np.ndarray | None = None,
    floored: np.ndarray | None = None,
    limit: int | None = None,
) -> tuple[Path, Path]:
    """Write a per-sample CSV report and a JSON summary for later inspection.

    The CSV has one row per extracted sample with labels, predictions, and
    per-class logits (logit_A, logit_B, ...). The JSON summary holds
    aggregate accuracy and floor statistics.
    """
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = report_path.with_name(report_path.stem + "_summary.json")

    n_samples = len(dataset.train)
    n_report = min(limit, n_samples) if limit else n_samples
    if extracted_mask is None:
        extracted_mask = np.ones(n_samples, dtype=bool)
    if floored is None:
        floored = logits == LOGPROB_FLOOR

    verbalizers = make_verbalizers(dataset.num_classes)
    logit_cols = [f"logit_{v}" for v in verbalizers]

    rows: list[dict[str, object]] = []
    for idx in range(n_report):
        if not extracted_mask[idx]:
            continue

        sample = dataset.train[idx]
        text = sample[dataset.text_column]
        true_label = int(sample[dataset.label_column])
        row_logits = logits[idx]
        pred_label = int(row_logits.argmax())
        num_floored = int(floored[idx].sum())

        row: dict[str, object] = {
            "sample_idx": idx,
            "true_label_id": true_label,
            "true_label_name": dataset.label_names[true_label],
            "pred_label_id": pred_label,
            "pred_label_name": dataset.label_names[pred_label],
            "correct": int(pred_label == true_label),
            "max_logit": float(row_logits.max()),
            "num_floored_classes": num_floored,
            "text": text,
        }
        for verbalizer, logit in zip(verbalizers, row_logits):
            row[f"logit_{verbalizer}"] = float(logit)
        rows.append(row)

    fieldnames = [
        "sample_idx",
        "true_label_id",
        "true_label_name",
        "pred_label_id",
        "pred_label_name",
        "correct",
        "max_logit",
        "num_floored_classes",
        "text",
        *logit_cols,
    ]
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        floored_subset = np.array([floored[r["sample_idx"]] for r in rows])
        floor_frac = float(floored_subset.sum() / floored_subset.size)
        accuracy = float(np.mean([r["correct"] for r in rows]))
    else:
        floor_frac = 0.0
        accuracy = 0.0

    summary = {
        "teacher": teacher_name,
        "dataset": dataset.name,
        "n_samples_reported": len(rows),
        "accuracy": accuracy,
        "floor_fraction": floor_frac,
        "report_csv": str(report_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return report_path, summary_path
