import openai
import anthropic




anthropic_client = anthropic.Anthropic(api_key="")
openai_client = openai.OpenAI(api_key="")


"""
Run Claude model

Args:
    prompt (string): The complete prompt with all required information for the task
    model (string): The Claude model to use. Default is Claude 3.5 Haiku

Returns:
    string: The output of the Claude model with respect to the prompt

"""
def run_claude(prompt: str, model="claude-3-5-haiku-20241022"):
    message = anthropic_client.messages.create(
        model=model,
        max_tokens=1000,
        temperature=0.7,
        system="Respond only in plain text. Do not use Markdown formatting. After you give the final answer do not continue.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    model_output = message.content[0].text
    return model_output




"""
Run OpenAI Model

Args:
    prompt (string): The complete prompt with all required information for the task
    model (string): The OpenAI model to use. Default is gpt-4o-mini

Returns:
    string: The output of the OpenAI model with respect to the prompt

"""
def run_gpt(prompt: str, model="gpt-4o-mini"):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Respond only in plain text. Do not use Markdown formatting. After you give the final answer do not continue."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature = 0.7
        
    )
    output = response.choices[0].message.content
    return output



"""
Run Anthropic and OpenAI models on a task

Args:
    prompt (string): The complete prompt with all required information for the task
    runs (int): The number of times to run each model(total runs = 2*runs)

Returns:
    list: A list of dictionaries containing the model, run number, its output

"""
def run_hybrid(prompt: str, runs=5):
    model_runs = []
    for i in range(runs):
        gpt_response = run_gpt(prompt)
        claude_response = run_claude(prompt)
        gpt_dict = {"model": "gpt", "run": i+1, "output": gpt_response}
        claude_dict = {"model": "claude", "run": i+1, "output": claude_response}
        model_runs.append(gpt_dict)
        model_runs.append(claude_dict)

    return model_runs


"""
Run Anthropic and OpenAI models on a task

Args:
    prompt (string): The complete prompt with all required information for the task
    runs (int): The number of times to run the model
    model(str): The model to run - options are (1)"openai" (2)"claude"

Returns:
    list: A list of dictionaries containing the model, run number, its output

"""
def run_single(prompt: str, runs=10, model="openai"):

    model_runs = []

    if(model=="openai"):
        for i in range(runs):
            gpt_response = run_gpt(prompt)
            gpt_dict = {"model": "gpt", "run": i+1, "output": gpt_response}
            model_runs.append(gpt_dict)
    elif(model=="claude"):
        for i in range(runs):
            claude_response = run_claude(prompt)
            claude_dict = {"model": "claude", "run": i+1, "output": claude_response}
            model_runs.append(claude_dict)
    else:
        print("Please choose a valid model to run(openai/claude)")

    return model_runs


"""
Run a simple test to check anthropic and openai APIs are working

Args:
    prompt (string): test prompt to run

Returns:
    list: output of both models

"""
def test(prompt: str):
    gpt_output = run_gpt(prompt)
    claude_output = run_claude(prompt)
    return gpt_output, claude_output
