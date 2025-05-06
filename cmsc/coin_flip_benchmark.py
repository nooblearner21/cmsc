import json
import re
import os, sys
import datasets
from collections import Counter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Insert it into sys.path
sys.path.insert(0, parent_dir)
import model



def run_benchmark():
    correct_answers = []
    wrong_answers = []
    total_correct = 0
    num_samples = 0
    with open("./benchmark_datasets/coin_flip_4_times_ood.json", "r") as file:
        for line in file:
            sample = json.loads(line)
            prompt = sample['inputs']
            output = sample['targets']

            model_answer = get_model_answers(prompt)

            if(model_answer == output):
                correct_answers.append({"model_answer": model_answer, "ground_trurth": output, "prompt": prompt})
                total_correct += 1
            else:
                print(f"Wrong answer {model_answer} for prompt:\n {prompt}")
                wrong_answers.append({"model_answer": model_answer, "ground_trurth": output, "prompt": prompt})

            num_samples += 1
            print(f"Finished Instance: {num_samples}")

    accuracy = (total_correct/num_samples) * 100
    print(f"The model ran on {num_samples} samples and correctly answered {total_correct} instances for a total accuracy of {accuracy:.2f}%")
    correct_answers.append("----------------SEPERATOR CORRECT/WRONG OUTPUTS----------------")
    correct_answers += wrong_answers
    #Write correct and wrong answers to file
    with open("benchmark_results/coin_flip_4_results_hybrid_10_runs.json", "w") as f:
        json.dump(correct_answers, f, indent=4)


def get_model_answers(prompt: str):
    model_outputs = model.run_hybrid(prompt, runs=5)
    model_answers = []
    for i in range(len(model_outputs)):
        try:
            extract_model_answer = re.findall(r"The answer is (\w+)\.", model_outputs[i]['output'])[-1]
            model_answers.append(extract_model_answer)
        except:
            extract_model_answer = None
    final_answer = best_answer(model_answers)
    return final_answer


def best_answer(answers: list):
    most_common_answer, count = Counter(answers).most_common(1)[0]
    print(f"The most recurring string is '{most_common_answer}' with {count} occurrences.")
    return most_common_answer

run_benchmark()