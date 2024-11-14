# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Predict with LLM on task."""
import os
import argparse
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))
from collections import defaultdict
import sys
sys.path.append(root_path)
sys.path.append(f"{root_path}/sem_uncertainty")
from semantic_entropy.utils import batch_llm_metric
from semantic_entropy.huggingface_models import HuggingfaceModel
from src.eval_utils import VLLM, get_available_servers

import json
import jsonlines
import pandas as pd
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--entailment_model', type=str, default="")
    parser.add_argument('--prompt_type', type=str,)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--max_alpha', type=float, default=1.0)
    parser.add_argument("--use_predicted", type=int, default=0)
    parser.add_argument("--iti_method", type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--str_process_layers', type=str, default="")
    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    max_alpha = args.max_alpha
    iti_method = args.iti_method
    model_name = args.model_name
    prompt_type = args.prompt_type
    str_process_layers = args.str_process_layers
    batch_size = args.batch_size
    print(f"Running refusal rate for {dataset} {split}")

    server_dict = get_available_servers()["meta-llama/Llama-3.1-70B-Instruct"]
    server_urls = server_dict["server_urls"]
    if "http" not in args.entailment_model and len(args.entailment_model) < 3:
        args.entailment_model = server_urls[int(args.entailment_model)]

    if args.use_predicted:
        output_base_dir = f"{root_path}/calibration/predicted_outputs/{dataset}/{model_name}/{prompt_type}/{split}"
    else:
        output_base_dir = f"{root_path}/calibration/outputs/{dataset}/{model_name}/{prompt_type}/{split}"
    
    if iti_method in [0, 2, 3, 4]:
        input_path = f'{output_base_dir}/with_vufi_{iti_method}_{str_process_layers}_{max_alpha}.jsonl'
    elif iti_method == 1:
        input_path = f'{output_base_dir}/with_vufi_{iti_method}_trivia_qa_{str_process_layers}_{max_alpha}.jsonl'
    results_fn = input_path.replace("with_vufi", 'acc')[:-1]
    print(f"Results will be saved to {results_fn}")
    
    all_acc = defaultdict(list) # {qid: [acc1, acc2, ...]}
    if os.path.exists(results_fn):
        with open(results_fn, 'r') as f:
            all_acc = json.load(f)
            all_acc = defaultdict(list, all_acc)
            
    data = pd.read_csv(f"{root_path}/datasets/{dataset}/{model_name}/{split}.csv")
    with jsonlines.open(input_path, 'r') as f:
        lines = list(f)
        assert len(lines) <= len(data)

    all_predicted_answers, all_example, all_qid = [], [], []
    for i, example in data.iterrows():
        if i >= len(lines):
            break
        qid = example['id']
        assert example['question'] == lines[i]["question"]
        if qid not in all_acc:
            all_example.append(example)
            all_qid.append(qid)
            a = lines[i]["most_likely_answer"]
            if a:
                all_predicted_answers.append(a)
            else:
                all_predicted_answers.append("None")
    print(f"len(all_predicted_answers)", len(all_predicted_answers))
    if len(all_predicted_answers):
        if 'http' in args.entailment_model:
            evaluator_model = VLLM(args.entailment_model, max_new_tokens=50)
        else:
            evaluator_model = HuggingfaceModel("Meta-Llama-3.1-70B-Instruct", max_new_tokens=50)
    for i in tqdm(range(0, len(all_predicted_answers), batch_size)):
        batch_predicted_answers = all_predicted_answers[i:i+batch_size]
        batched_example = all_example[i:i+batch_size]
        batched_qid = all_qid[i:i+batch_size]
        
        acces = batch_llm_metric(batch_predicted_answers, batched_example, evaluator_model, prompt_type='ignore_vu')
        assert len(acces) == len(batch_predicted_answers) == len(batched_example) == len(batched_qid)
        for j, acc in enumerate(acces):
            all_acc[batched_qid[j]].append(acc)

        if i % (batch_size*10) == 0:
            torch.cuda.empty_cache()
            print(f"Processed {i} examples")
            with open(results_fn, 'w') as f:
                json.dump(all_acc, f)

    with open(results_fn, 'w') as f:
        json.dump(all_acc, f)