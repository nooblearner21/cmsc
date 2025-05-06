import json
from collections import Counter

def count_matching_top_answers(json_path):
    """
    Parses the JSON file at json_path, computes the most frequent answer
    for 'gpt' and 'claude' in each entry, and counts how many entries
    have the same top answer for both models.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    match_count = 0
    
    for entry in data:
        # Extract raw outputs
        raw = entry.get('parsed_answers', {})
        gpt_answers = raw.get('gpt', [])
        claude_answers = raw.get('claude', [])
        
        # Skip if either list is empty
        if not gpt_answers or not claude_answers:
            continue
        
        # Determine most common answer for each
        gpt_top, _ = Counter(gpt_answers).most_common(1)[0]
        claude_top, _ = Counter(claude_answers).most_common(1)[0]
        
        # Compare and increment counter if they match

        if gpt_top != claude_top:
            print(f"gpt:{ gpt_top}, claude: {claude_top}")
            match_count += 1
    
    return match_count

if __name__ == "__main__":
    json_file_path = 'svamp_log_hybrid_500_fixed.json'  # Update with your JSON file path
    matches = count_matching_top_answers(json_file_path)
    print(f"Number of entries where GPT and Claude share the same top answer: {matches}")
