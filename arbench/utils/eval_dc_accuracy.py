import json
import sys
from typing import List
from arbench.utils.utils_dc import ANSWER_CHOICES, calculate_accuracy


def load_results(result_file: str):
    with open(result_file, "r", encoding="utf-8") as f:
        logs = json.load(f)
    preds = [entry["pred"] for entry in logs]
    labels = [entry["label"] for entry in logs]
    return preds, labels


def normalize_label(x):
    """Convert label/pred to index [0..3], or None if invalid."""
    if isinstance(x, int):
        if 0 <= x < len(ANSWER_CHOICES):
            return x
    if isinstance(x, str):
        x = x.strip().upper()
        if x in ANSWER_CHOICES:
            return ANSWER_CHOICES.index(x)
    return None


def print_confusion_matrix(preds: List, labels: List):
    n = len(ANSWER_CHOICES)
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    skipped = 0

    for p, l in zip(preds, labels):
        p_idx = normalize_label(p)
        l_idx = normalize_label(l)
        if p_idx is None or l_idx is None:
            skipped += 1
            continue
        matrix[l_idx][p_idx] += 1

    header = "Pred→ | " + " | ".join([f"{c:^5}" for c in ANSWER_CHOICES])
    print("\nConfusion Matrix (rows = true labels, cols = predictions):")
    print(header)
    print("-" * len(header))
    for i, row in enumerate(matrix):
        row_str = " | ".join([f"{v:^5}" for v in row])
        print(f" True {ANSWER_CHOICES[i]} | {row_str}")

    if skipped > 0:
        print(f"\n[!] Skipped {skipped} samples due to invalid labels/preds")


def summarize(result_file: str):
    preds, labels = load_results(result_file)
    acc = calculate_accuracy(preds, labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    total = len(labels)

    print(f"Results from {result_file}")
    print(f"Cases: {total}")
    print(f"Accuracy (same as calculate_accuracy): {acc:.4f} ({correct}/{total})")

    print_confusion_matrix(preds, labels)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m arbench.utils.eval_dc_accuracy <result_file.json>")
    else:
        summarize(sys.argv[1])
