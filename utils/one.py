#!/usr/bin/env python3
import json
import argparse
import sys

def load_json_entries(path):
    """
    Read a file that may be:
      - A JSON array (possibly with trailing commas)
      - Multiple JSON objects back-to-back (with or without commas)
    and return a flat list of Python dicts.
    """
    decoder = json.JSONDecoder()
    # open with utf-8-sig so a BOM (if present) is stripped
    with open(path, 'r', encoding='utf-8-sig') as f:
        text = f.read()

    # Try the easy path first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        else:
            return [data]
    except json.JSONDecodeError:
        pass

    objs = []
    idx = 0
    length = len(text)
    while idx < length:
        # skip whitespace, commas, and array brackets
        while idx < length and (text[idx].isspace() or text[idx] in ',[]'):
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
        description="Convert a JSON array or concatenated objects into JSON-Lines"
    )
    parser.add_argument("input_file", help="Path to your mixed-format JSON file")
    parser.add_argument("output_file", help="Where to write the JSON-Lines output")
    args = parser.parse_args()

    entries = load_json_entries(args.input_file)

    with open(args.output_file, 'w', encoding='utf-8') as out:
        for e in entries:
            out.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {len(entries)} JSON objects to {args.output_file} (one per line).")

if __name__ == "__main__":
    main()
