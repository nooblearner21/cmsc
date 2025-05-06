import json
import re
import sys
import os
from datetime import datetime
from collections import Counter, defaultdict

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import model

def run_model(prompt: str):
    # returns 10 outputs
    return model.run_hybrid(prompt, runs=5)

def best_answer(answers):
    return Counter(answers).most_common(1)[0][0]

def extract_answers(model_outputs):
    """Return both the raw answers (strings) and the parsed integers, skipping failures."""
    answers_raw, answers_parsed = [], []
    for out in model_outputs:
        ans_match = re.findall(r"The answer is\s+(\d+)", out["output"])
        answers_raw.append(out["output"])
        if ans_match:
            answers_parsed.append(ans_match[-1])
    return answers_raw, answers_parsed

def group_by_model(outputs):
    """
    From a tuple of {'model': ..., 'output': ...} dicts, produce
    {model_name: [raw_output1, raw_output2, ...]}
    and a parallel dict of parsed numeric answers.
    """
    raw = defaultdict(list)
    parsed = defaultdict(list)

    for item in outputs:
        m = item["model"]
        raw[m].append(item["output"])

        match = re.findall(r"The answer is\s+(\d+)", item["output"])
        if match:
            parsed[m].append(match[-1])

    return raw, parsed

def benchmark_svamp(json_path: str, *, slice_n=None, log_path="svamp_log.json", tol=1e-3):
    with open(json_path, "r") as fp:
        data = json.load(fp)

    if slice_n:
        data = data[:slice_n]          # e.g. slice_n=5 for quick tests
    print(len(data), "examples")

    total, correct = len(data), 0
    few_shot = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\n"

    results_log = []                   # full run log

    for ex in data:
        prompt = few_shot + f"{ex['Body']}\n{ex['Question']}"
        outs = run_model(prompt)

        raw_by_model, parsed_by_model = group_by_model(outs)

        # combine all parsed answers (across all models) to choose majority
        majority_pool = [a for lst in parsed_by_model.values() for a in lst]
        final_answer = best_answer(majority_pool) if majority_pool else None

        if final_answer is not None and abs(float(final_answer) - ex["Answer"]) < tol:
            correct += 1

        # ---- per-example log record --------------------------------------
        results_log.append({
            "id":            ex["ID"],
            "prompt":        prompt,
            "ground_truth":  ex["Answer"],
            "raw_outputs":   raw_by_model,      # {model: [str, str, ...]}
            "parsed_answers": parsed_by_model,  # {model: [ "13", "15", ... ]}
            "chosen_answer": final_answer
        })

    # save once at the end
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(results_log, fp, indent=2)
    print(f"saved detailed log to {log_path}")

    return correct / total if total else 0.0

# ---------------------------
# CLI entry point
# ---------------------------
if __name__ == "__main__":
    json_filepath = "benchmark_datasets/svamp.json"
    avg_accuracy = benchmark_svamp(json_filepath, slice_n=500)
    print(f"Average accuracy: {avg_accuracy:.2%}")
