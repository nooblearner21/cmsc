#!/usr/bin/env python3
import json
import argparse
import sys

def load_concatenated_json(path):
    """
    Read a file containing multiple JSON objects back-to-back
    and return them as a Python list.
    """
    decoder = json.JSONDecoder()
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    objs = []
    idx = 0
    length = len(text)
    while idx < length:
        # Skip whitespace
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break

        try:
            obj, offset = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON at char {idx}: {e}", file=sys.stderr)
            sys.exit(1)

        objs.append(obj)
        idx += offset

    return objs

def count_matches(objs):
    """
    Count how many objects have model_answer == ground_truth.
    Supports both 'ground_trurth' (typo) and 'ground_truth'.
    """
    count = 0
    for o in objs:
        model_ans = o.get("model_answer")
        # handle possible key typo
        ground = o.get("ground_truth", o.get("ground_trurth"))
        if model_ans is not None and ground is not None and model_ans == ground:
            count += 1
    return count

def main():
    parser = argparse.ArgumentParser(
        description="Count how many entries have model_answer == ground_truth"
    )
    parser.add_argument("json_file", help="Path to the JSON file")
    args = parser.parse_args()

    all_objs = load_concatenated_json(args.json_file)
    total = len(all_objs)
    matched = count_matches(all_objs)

    print(f"Total entries: {total}")
    print(f"Matches (model_answer == ground_truth): {matched}")

if __name__ == "__main__":
    main()
