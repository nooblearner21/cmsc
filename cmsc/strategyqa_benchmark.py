import json
import random
from collections import Counter
import os, sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Insert it into sys.path
sys.path.insert(0, parent_dir)
import model


def load_benchmark(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data

def run_hybrid(prompt: str, runs=5):

    return model.run_hybrid(prompt, 1)
    outputs = []
    for i in range(runs):
        outputs.append({'model': 'gpt', 'run': i+1, 'output': 'The answer is yes.'})
        outputs.append({'model': 'claude', 'run': i+1, 'output': 'The answer is no.'})
    return outputs

def run_single(prompt: str, runs=5, model="openai"):
    outputs = []
    for i in range(runs):
        output_text = 'The answer is yes.' if random.random() > 0.5 else 'The answer is no.'
        outputs.append({'model': model, 'run': i+1, 'output': output_text})
    return outputs

def eval_query(model, query, reps=5):
    few_shot_string = """Q: Yes or no: Is it common to see frost during some college commencements?
A: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. The answer is yes.

Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?
A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen's atomic number squared is less than 5. The answer is no.

Q: Could Brooke Shields succeed at University of Pennsylvania?
A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. The answer is yes.

Q: Yes or no: Would a pear sink in water?
A: The density of a pear is about 0.6 g/cm^3, which is less than water. Objects less dense than water float. Thus, a pear would float. The answer is no.

Q: Do hamsters provide food for any animals?
A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. The answer is yes.

Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?
A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. The answer is no.

Q: Yes or no: """

    question = few_shot_string + query['question']

    if model == 'hybrid':
        model_outputs = run_hybrid(question, reps)
    else:
        model_outputs = run_single(question, reps, model)

    responses = []
    for output in model_outputs:
        text = output['output'].strip().lower()
        answer_token = text.split()[-1]
        if answer_token not in ['yes.', 'no.']:
            print(f"Invalid response: {text}")
            continue
        is_yes = True if answer_token == 'yes.' else False
        correct = is_yes == query['answer']
        responses.append({'model response': is_yes, 'correct': correct, 'answer': query['answer']})
    return responses

def aggregate_responses(responses):
    if not responses:
        return 0, False
    counts = Counter(response['model response'] for response in responses)
    most_common = counts.most_common(1)[0]
    confidence = most_common[1] / len(responses) * 100
    final_answer = most_common[0]
    correct = final_answer == responses[0]['answer']
    return confidence, correct

def evaluate(model, path, reps):
    data = load_benchmark(path)
    queries = []
    correct = 0

    for query in data:
        responses = eval_query(model, query, reps)
        confidence, is_correct = aggregate_responses(responses)
        queries.append({
            'question': query['question'],
            'model confidence': confidence,
            'correct': is_correct,
            'answer': query['answer']
        })
        if is_correct:
            correct += 1
        if(correct == 3):
            break
    accuracy = correct / len(queries) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Example call:
evaluate('hybrid', 'benchmark_datasets/strategyqa.json', 1)
