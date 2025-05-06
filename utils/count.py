#!/usr/bin/env python3
"""
Quick accuracy checker for AQuA‑style result files.

Each JSON entry is expected to contain:
    {
        "id": …,               # any identifier
        "question": …,         # (unused here but nice to keep)
        "model_answer": "B",
        "ground_truth": "B"
    }
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict


def normalise(ans: str) -> str:
    """Case‑insensitive, strip whitespace & trailing punctuation."""
    return ans.strip().rstrip(".").lower()


def load(path: Path) -> List[Dict]:
    with path.open(encoding="utf‑8") as f:
        return json.load(f)


def compute_accuracy(records: List[Dict]) -> float:
    correct = sum(
        1 for r in records
        if normalise(r["model_answer"]) == normalise(r["ground_trurth"])
    )
    return correct / len(records) if records else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute answer accuracy for AQuA result files."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to JSON file with model results"
    )
    args = parser.parse_args()

    try:
        data = load(args.file)
    except FileNotFoundError:
        print(f"❌ File not found: {args.file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ JSON load error: {e}", file=sys.stderr)
        sys.exit(1)

    acc = compute_accuracy(data)
    correct = int(acc * len(data))
    total = len(data)
    print(f"✅ {correct}/{total} correct  |  accuracy = {acc:.2%}")

    # return 0 only if all answers are correct
    sys.exit(0 if acc == 1.0 else 1)


if __name__ == "__main__":
    main()
