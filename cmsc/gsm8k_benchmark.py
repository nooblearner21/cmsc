"""
Benchmark GSM8K with grouped-by-model logging.

▪ Requires  model.run_hybrid(prompt, 5)  to return a tuple of 10 dicts like
  {'model': 'gpt', 'run': 1, 'output': 'The answer is 13'}

▪ Produces gsm8k_run_YYYYMMDD_HHMMSS.json  with detailed per-example info.
"""
import os
import sys
import json
import re
from datetime import datetime
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
#  Plumbing – import your model package
# --------------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import model  # noqa: E402

# --------------------------------------------------------------------------
#  Regex & helpers
# --------------------------------------------------------------------------
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def extract_gt_answer(text: str):
    """Return int ground-truth from GSM8K answer field."""
    m = ANS_RE.search(text)
    if not m:
        return INVALID_ANS
    return int(m.group(1).replace(",", ""))


def run_model(prompt: str):
    """Call your self-refine model: returns tuple of 10 dicts."""
    return model.run_hybrid(prompt, runs=5)
    #return model.run_hybrid(prompt, runs=5)

# --------------------------------------------------------------------------
#  Output processing
# --------------------------------------------------------------------------
def group_by_model(outputs):
    """
    outputs: tuple[dict]  →  (raw_by_model, parsed_by_model)

    raw_by_model    = {model_name: [full_output_str, ...]}
    parsed_by_model = {model_name: ['13', '15', ...]}  (regex-matched numbers)
    """
    raw, parsed = defaultdict(list), defaultdict(list)
    for item in outputs:
        mname = item["model"]
        text = item["output"]
        raw[mname].append(text)

        # OLD
        #nums = re.findall(r"The answer is\s+(\d+)", text)

        # NEW ─ allows digits, optional commas, optional leading minus and an optional decimal part
        nums = re.findall(r"The answer is\s+([\-]?\d[\d,]*\.?\d*)", text)

        if nums:
            clean_number = nums[-1].replace(",", "")
            parsed[mname].append(clean_number)  # use last match in string
    return raw, parsed


def majority_vote(flat_answers):
    """Return the most common answer string (ties broken arbitrarily)."""
    return Counter(flat_answers).most_common(1)[0][0] if flat_answers else None


# --------------------------------------------------------------------------
#  Benchmark core
# --------------------------------------------------------------------------
def benchmark_gsm8k(jsonl_path: str, *,
                    slice_n: int | None = None,
                    log_path: str | None = None,
                    tolerance: float = 1e-3) -> float:
    # -------- dataset --------
    with open(jsonl_path, encoding="utf-8") as fp:
        data = [json.loads(l) for l in fp]
    if slice_n:
        data = data[:slice_n]

    # -------- log file name --------
    if log_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"gsm8k_run_{ts}.json"

    run_log = []
    correct = 0

    # -------- few-shot block (unchanged from your code) --------
    FEW_SHOT = "Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\n"


    # -------- main loop --------
    for ex in data:
        prompt = FEW_SHOT + ex["question"] + "\n"

        outputs = run_model(prompt)
        raw_by_model, parsed_by_model = group_by_model(outputs)

        pool = [num for lst in parsed_by_model.values() for num in lst]
        prediction = majority_vote(pool)

        gt = extract_gt_answer(ex["answer"])
        if prediction is not None and gt != INVALID_ANS:
            num_pred = int(float(prediction.replace(",", "")))  # <- new
            if prediction is not None and gt != INVALID_ANS and abs(num_pred - gt) < tolerance:
                correct += 1

        # ----- save record -----
        run_log.append({
            "id": ex.get("id", ex.get("idx", None)),
            "prompt": prompt,
            "ground_truth": gt,
            "raw_outputs": raw_by_model,       # {model: [str, ...]}
            "parsed_answers": parsed_by_model, # {model: ['13', ...]}
            "chosen_answer": prediction
        })

    # -------- dump log file --------
    with open(log_path, "w", encoding="utf-8") as fp:
        json.dump(run_log, fp, indent=2)
    print(f"[log] saved detailed run to '{log_path}'")

    return correct / len(data) if data else 0.0


# --------------------------------------------------------------------------
#  CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_FILE = "./benchmark_datasets/gsm8k.json"   # path to your .jsonl
    acc = benchmark_gsm8k(DATA_FILE,slice_n=500)
    print(f"Average accuracy on GSM8K: {acc:.2%}")
