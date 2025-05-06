import json
import re

import sys, os
import datasets
from collections import Counter

from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Insert it into sys.path
sys.path.insert(0, parent_dir)
import model

benchmark_name = "aqua"
model_type = "gpt"

def run_benchmark():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    correct_answers = []
    wrong_answers = []
    total_correct = 0
    num_samples = 0
    counter = 0
    with open("./benchmark_results/missing_gpt.json", "r", encoding='utf-8', errors='replace') as file:
        for line in file:
            sample = json.loads(line)
            prompt = sample['input']
            output = sample['answer']
            model_answer, gpt_answers, claude_answers, raw_outputs = get_model_answers(prompt)
            if(model_answer == None):
                continue
            print(model_answer)
            print(output)
            if(model_answer == output):
                total_correct += 1
            else:
                print(f"Wrong answer {model_answer} for prompt:\n ")
            num_samples += 1
            print(f"Finished Instance: {num_samples}")
            with open(f"benchmark_results/{benchmark_name}_results_{model_type}_10_runs_{timestamp}.json", "a") as f:
                json.dump({"model_answer": model_answer, "ground_trurth": output, "prompt": prompt, "raw_outputs": raw_outputs, 'gpt_answers': gpt_answers, "claude_answers": claude_answers}, f, indent=4)
    accuracy = (total_correct/num_samples) * 100
    print(f"The model ran on {num_samples} samples and correctly answered {total_correct} instances for a total accuracy of {accuracy:.2f}%")

    #Write correct and wrong answers to file




def get_model_answers(prompt: str):
    prompt = "No matter what, give an answer at the end 'The answer is ' with your final answer.\n" + prompt + "\nA: "
    model_outputs = model.run_single(prompt, model="openai")
    gpt_answers = []
    claude_answers = []
    model_answers = []
    pattern = re.compile(
    r"\bthe answer is\s*\(\s*([A-Za-z])\s*\)",
    re.IGNORECASE
)
    for i in range(len(model_outputs)):
        try:
            extract_model_answer = pattern.findall( model_outputs[i]['output'])[-1]
            extract_model_answer = f"({extract_model_answer})"
            model_answers.append(extract_model_answer)
            if(model_outputs[i]['model'] == 'gpt'):
                gpt_answers.append(extract_model_answer)
            else:
                claude_answers.append(extract_model_answer)
        except:
            extract_model_answer = None
            print(i)
            print(model_outputs)
    final_answer = best_answer(model_answers)
    if(final_answer == None):
        return None, None, None, None
    return final_answer, gpt_answers, claude_answers, model_outputs


def best_answer(answers: list):
    try:
        most_common_answer, count = Counter(answers).most_common(1)[0]
    except:
        return None
    print(f"The most recurring string is '{most_common_answer}' with {count} occurrences.")
    return most_common_answer

run_benchmark()