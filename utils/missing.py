#!/usr/bin/env python3
import json
import argparse
import sys

def load_json_entries(path):
    """
    Load JSON entries from a file. Supports:
    - A single JSON array (loads all elements)
    - A single JSON object (returns [object])
    - Multiple JSON objects back-to-back (no commas)
    """
    decoder = json.JSONDecoder()
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Try loading as a single JSON value
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        else:
            return [data]
    except json.JSONDecodeError:
        pass

    # Fallback: parse concatenated JSON objects
    objs = []
    idx = 0
    length = len(text)
    while idx < length:
        # skip whitespace
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


def main():
    parser = argparse.ArgumentParser(
        description="Find prompts in file1 not in file2, output input-answer pairs"
    )
    parser.add_argument("file1", help="Path to the first JSON file")
    parser.add_argument("file2", help="Path to the second JSON file")
    parser.add_argument("output_file", help="Path to write missing entries JSON array")
    args = parser.parse_args()

    # Load entries
    entries1 = load_json_entries(args.file1)
    entries2 = load_json_entries(args.file2)

    # Build set of prompts in second file
    prompts2 = {e.get("prompt") for e in entries2 if "prompt" in e}

    # Collect missing entries with input & answer
    missing = []
    for e in entries1:
        prompt = e.get("prompt")
        if prompt and prompt not in prompts2:
            # support both ground_truth keys
            answer = e.get("ground_truth", e.get("ground_trurth"))
            missing.append({
                "input": prompt,
                "answer": answer
            })

    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(missing, f, indent=2, ensure_ascii=False)

    print(f"Found {len(missing)} missing entries and saved to {args.output_file}")

if __name__ == "__main__":
    main()
