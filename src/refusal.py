import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
import json
from tqdm.contrib.concurrent import thread_map
import openai
import sys
sys.path.append(root_path)
from src.utils import init_model
import os
import json
import torch
from tqdm import tqdm
from typing import List


ABSTAIN_PROMPT = """You are given a prompt containing a question and a generation by an AI. Your task is to determine whether the AI refuse to answer the question or not.

Return your result as valid JSON using: {{"does_refuse": true|false}}

Some previous examples are below:
Example 1:
Question: What type of creature is a Meadow Brown?
Generation: Butterfly
Answer: {{"does_refuse": false}}

Example 2:
Question: Who is a South African serial killer who committed the ""ABC Murders"" between 1994 and 1995, with over 30 victims?
Generation: I cannot verify who the South African serial killer was that committed the "ABC Murders" between 1994 and 1995.
Answer: {{"does_refuse": true}}

Example 3:
Question: From which musical film does the song ""Wooden Heart"" come?
Generation: The song "Wooden Heart" comes from the musical film "Blue Hawaii"
Answer: {{"does_refuse": false}}

Example 4:
Question: After which war did the Ghadar Party in the United States split into different factions, and what were those factions?
Generation: I can't answer that. I don't have information on the Ghadar Party in the United States splitting into factions after a war.
Answer: {{"does_refuse": true}}

Example 5:
Question: What was the duration of the Acacian schism, and during which years did it occur?
Generation: I can't verify the duration of the Acacian schism and the years it occurred.
Answer: {{"does_refuse": true}}

According to Generation, does AI refuse to answer the question?
Question: {question}
Generation: {generation}
Answer: 
"""



def batch_predict(tokenizer, model, batch_generations):
    batch_prompts = []
    for generation in batch_generations:
        abstain_prompt = ABSTAIN_PROMPT.format(
                question=generation['question'], 
                generation=generation['answer'],
            )
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": abstain_prompt},], tokenize=False, add_generation_prompt=True)
        batch_prompts.append(prompt)

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    initial_length = inputs['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=0.1,
            do_sample=True,
        )
    
    batch_answer = tokenizer.batch_decode(outputs.sequences[:, initial_length:], skip_special_tokens=True)
    return batch_answer

def automatic_abstention(generations, model, tokenizer):
    if type(model) == str:
        abstain_prompts = [
            ABSTAIN_PROMPT.format(
                question=generation['question'], 
                generation=generation['answer'],
            )
            for generation in generations
        ]
        abstains_eval_raw = thread_map(
            lambda p: call_vllm_api(p, port=model),
            abstain_prompts,
            max_workers=20,
            desc="using vllm")

    else:
        batch_size = 16
        abstains_eval_raw = []
        for i in tqdm(range(0, len(generations), batch_size)):
            batch_generations = generations[i:i+batch_size]
            batch_answer = batch_predict(tokenizer, model, batch_generations)
            abstains_eval_raw += batch_answer

    abstains_eval = jsonify_ans(raw_responses=abstains_eval_raw, key="does_refuse")
    abstains_eval_res = []
    for o in abstains_eval:
        try:
            abstains_eval_res.append(o['does_refuse'])
        except:
            print(f"Error in eval_answer: {o}")
            exit()
    assert len(abstains_eval_res) == len(generations)
    return abstains_eval_res

def call_vllm_api(prompt, port=None):
    client = openai.OpenAI(
        # base_url=f"http://{port}/v1",
        base_url=f"{port}",
        api_key="NOT A REAL KEY",
    )
    chat_completion = client.chat.completions.create(
        model='meta-llama/Llama-3.1-70B-Instruct',
        messages=[{"role": "user","content": prompt}],
        max_tokens=512,
        temperature=0.1,
        top_p=0.9,
    )

    return chat_completion.choices[0].message.content


def jsonify_ans(raw_responses: List[str], key: str):
    
    def check_validity(gen):
        gen_nospace = gen.replace(" ", "")
        if '{{"{}":false}}'.format(key) in gen_nospace:
            return '{{"{}":false}}'.format(key)
        elif '{{"{}":true}}'.format(key) in gen_nospace:
            return '{{"{}":true}}'.format(key)
        else:
            return -1
        
    jsonifyed_res  = []
    for r in raw_responses:
        if check_validity(r) != -1:
            jsonifyed_res.append(json.loads(check_validity(r)))
            continue
        else:
            r = r.split("\n")[0]
            jsonifyed_res.append(json.loads(r))
    return jsonifyed_res




def run_eval(generations, res_path, port="", overwrite=False):
    history_i = 0
    history_refusal = []
    if os.path.exists(res_path) and not overwrite:
        res = json.load(open(res_path, "r"))
        history_refusal = res["refusal"]
        history_i = len(history_refusal)
        if history_i == len(generations):
            print(f"Skipping {res_path}")
            return res
        generations = generations[history_i:]
        print('history_i', history_i, len(generations))

    if not port:
        model, tokenizer = init_model("Llama-3.1-70B-Instruct", "cuda", padding_side="left", load_model=True)
        model.eval()

    # save every 100
    for i in range(0, len(generations), 100):
        batch_generations = generations[i:i+100]
        if port:
            print("Using VLLM API")
            batch_eval_results = automatic_abstention(batch_generations, port, None)
        else:
            # load the model
            batch_eval_results = automatic_abstention(batch_generations, model, tokenizer)

        history_refusal += batch_eval_results

        res = {
            'refusal': history_refusal,
            'refusal_rate': sum(history_refusal)/len(history_refusal)
        }
        # save the results
        with open(res_path, 'w') as f:
            json.dump(res, f, indent=4)
    return res
