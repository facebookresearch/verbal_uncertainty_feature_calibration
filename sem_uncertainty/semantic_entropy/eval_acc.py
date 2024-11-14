"""Predict with LLM on task."""
import os
import argparse
from tqdm import tqdm

from collections import defaultdict
import jsonlines
import json
current_path = os.getcwd()
root_path = os.path.abspath(os.path.join(current_path, os.pardir))
import sys
sys.path.append(f'{root_dir}/sem_uncertainty/')
from utils.utils import batch_llm_metric
from models.huggingface_models import HuggingfaceModel
from tqdm.contrib.concurrent import thread_map

class VLLM:
    # HuggingfaceModel
    def __init__(self, name, max_new_tokens):
        self.name = 'meta-llama/Llama-3.1-70B-Instruct'
        self.port = name
        self.max_tokens = max_new_tokens

    def predict(self, prompt, temperature, output_hidden_states=False):
        client = openai.OpenAI(
            base_url=self.port,
            api_key="NOT A REAL KEY",
        )
        chat_completion = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user","content": prompt}],
            max_tokens=self.max_tokens,
            temperature=temperature,
        )

        return chat_completion.choices[0].message.content, '', ''
    
    def batch_predict(self, batch_prompts, temperature, output_hidden_states=False):
        all_return_values = thread_map(
            lambda p: self.predict(p, temperature=temperature),
            batch_prompts,
            max_workers=20,
            desc="using vllm")
        
        # all_return_values.append((sliced_answer, log_likelihoods, hidden_states))
        try:
            assert type(all_return_values) == list
            assert len(all_return_values[0]) == 3
            assert type(all_return_values[0][0]) == str
        except:
            print("batch_prompts", batch_prompts)
            print("all_return_values", type(all_return_values), all_return_values)
            assert False
        return all_return_values



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--type', type=str, default="sentence")
    parser.add_argument('--port', type=str, default="")
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()

    # evaluator_model = HuggingfaceModel("Meta-Llama-3.1-70B-Instruct", max_new_tokens=50)
    evaluator_model = VLLM(args.port, max_new_tokens=50)
    dataset = args.dataset
    split = args.split
    batch_size = 40
    print(f"Running refusal rate for {dataset} {split}")

    output_base_dir = f"{root_dir}/sem_uncertainty/outputs/{dataset}/{args.type}/{args.model_name}"
    res_path = f"{output_base_dir}/{split}_most_likely_acc.json"
    all_acc = {}
    if os.path.exists(res_path):
        with open(res_path, 'r') as f:
            all_acc = json.load(f)
    
    input_path = f"{output_base_dir}/{split}_0.1.jsonl"
    print('input_path', input_path)
    assert not os.path.isdir(input_path)
    with jsonlines.open(input_path) as f:
        all_predicted_answers, all_example, all_qid = [], [], []
        for example in f:
            qid = example['id']
            if qid in all_acc:
                continue
            responses = example["model answers"]
            assert len(responses) == 1
            response = responses[0]
            example["answer"] = example["answer"][:20]
            all_qid.append(qid)
            all_example.append(example)
            all_predicted_answers.append(response)
    print("len(all_example)", len(all_example))

    for i in tqdm(range(0, len(all_predicted_answers), batch_size)):
        batch_predicted_answers = all_predicted_answers[i:i+batch_size]
        batched_example = all_example[i:i+batch_size]
        batched_qid = all_qid[i:i+batch_size]

        acces = batch_llm_metric(batch_predicted_answers, batched_example, evaluator_model, prompt_type="default")
        assert len(acces) == len(batch_predicted_answers) == len(batched_example) == len(batched_qid)
        for qid, acc in zip(batched_qid, acces):
            all_acc[qid] = acc

        if i % batch_size*10 == 0:
            print(f"Processed {i} examples")
            with open(res_path, 'w') as f:
                json.dump(all_acc, f)

    with open(res_path, 'w') as f:
        json.dump(all_acc, f)