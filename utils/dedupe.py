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

def dedupe_by_prompt(objs):
    """
    Remove any object whose 'prompt' value has already been seen.
    Keeps the first occurrence of each unique prompt.
    """
    seen = set()
    unique = []
    for o in objs:
        p = o.get("prompt")
        if p not in seen:
            seen.add(p)
            unique.append(o)
    return unique

def main():
    parser = argparse.ArgumentParser(
        description="Load a concatenated-JSON file and remove entries with duplicate prompts."
    )
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to write deduplicated JSON array")
    args = parser.parse_args()

    all_objs = load_concatenated_json(args.input_file)
    deduped = dedupe_by_prompt(all_objs)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(deduped)} unique entries (out of {len(all_objs)}) "
          f"to {args.output_file}")

if __name__ == "__main__":
    main()
