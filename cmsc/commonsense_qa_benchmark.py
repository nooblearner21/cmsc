from datasets import load_dataset
import random
from collections import Counter
import os, sys
import re

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Insert it into sys.path
sys.path.insert(0, parent_dir)
import model

dataset = load_dataset("commonsense_qa")['validation']

def concatenate_choices(choices):
    returned = ""
    for i in range(len(choices['label'])):
        returned += f'({choices["label"][i].lower()}) {choices["text"][i].lower()}\n'
    return returned

def run_hybrid(prompt: str, runs=5):
    return model.run_hybrid(prompt, 1)
    outputs = []
    for i in range(runs):
        outputs.append({'model': 'gpt', 'run': i+1, 'output': 'The answer is (a).'})
        outputs.append({'model': 'claude', 'run': i+1, 'output': 'The answer is (a).'})
    return outputs

def run_single(prompt: str, runs=5, model="openai"):
    outputs = []
    for i in range(runs):
        output_text = 'The answer is yes.' if random.random() > 0.5 else 'The answer is no.'
        outputs.append({'model': model, 'run': i+1, 'output': output_text})
    return outputs

def eval_query(model, query, reps=5):
    few_shot_string = few_shot_string = """Q: Sammy wanted to go to where the people were. Where might he go?
Answer Choices:
(a) populated areas
(b) race track
(c) desert
(d) apartment
(e) roadblock
A: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. The answer is (a).

Q: The fox walked from the city into the forest, what was it looking for?
Answer Choices:
(a) pretty flowers
(b) hen house
(c) natural habitat
(d) storybook
A: Answer: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. The answer is (b).

Q: What home entertainment equipment requires cable?
Answer Choices:
(a) radio shack
(b) substation
(c) television
(d) cabinet
A: The answer must require cable. Of the above choices, only television requires cable. The answer is (c).

Q: Google Maps and other highway and street GPS services have replaced what?
Answer Choices:
(a) united states
(b) mexico
(c) countryside
(d) atlas
A: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. The answer is (d).

Q: What do people use to absorb extra ink from a fountain pen?
Answer Choices:
(a) shirt pocket
(b) calligrapher's hand
(c) inkwell
(d) desk drawer
(e) blotter
A: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. The answer is (e).

Q: Where do you put your grapes just before checking out?
Answer Choices:
(a) mouth
(b) grocery cart
(c)super market
(d) fruit basket
(e) fruit market
A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. The answer is (b).

Q: Before getting a divorce, what did the wife feel who was doing all the work?
Answer Choices:
(a) harder
(b) anguish
(c) bitterness
(d) tears
(e) sadness
A: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. The answer is (c).
"""

    question = few_shot_string + '\nQ: ' + query['question'] + "\nAnswer Choices:\n" + concatenate_choices(query['choices']) + "A:"
    print(question)

    if model == 'hybrid':
        model_outputs = run_hybrid(question, reps)
    else:
        model_outputs = run_single(question, reps, model)

    responses = []
    for output in model_outputs:
        text = output['output'].strip().lower()
        matches = re.findall(r"the answer is \((.)\)", text)
        if matches:
            answer_token = matches[-1]  # return the last match
        else:
            answer_token = None 

        print(answer_token)
        # answer_token = text.split()[-1]
        # answer_token = answer_token[-3]
        if answer_token not in ['a', 'b', 'c', 'd', 'e']:
            print(f"Invalid response: {text}")
            continue
        answer_token = answer_token.upper()
        correct = answer_token == query['answerKey']
        responses.append({'model response': answer_token, 'correct': correct, 'answer': query['answerKey']})

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

def evaluate(model, reps):
    data = load_dataset("commonsense_qa")['validation']
    queries = []
    correct = 0

    for query in data:
        responses = eval_query(model, query, reps)
        confidence, is_correct = aggregate_responses(responses)
        queries.append({
            'question': query['question'],
            'model confidence': confidence,
            'correct': is_correct,
            'answer': query['answerKey']
        })
        if is_correct:
            correct += 1

    accuracy = correct / len(queries) * 100
    print(f"Accuracy: {accuracy:.2f}%")

evaluate('hybrid', 15)
