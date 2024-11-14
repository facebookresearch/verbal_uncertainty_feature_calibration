import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
set_seed(42)
import pandas as pd
from tqdm.auto import tqdm
from os.path import join
import numpy as np
from ast import literal_eval
import argparse

import sys
import os
import json
import jsonlines
from uncertainty.utils import utils
current_path = os.getcwd()
root_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(root_path)
from verbal_uncertainty.qa_generate import load_qa_ds

def prepare_inputs(tokenizer, batch_local_prompt):
    batch_messages = []
    for p in batch_local_prompt:
        messages = [
            {"role": "user", "content": p},
        ]
        batch_messages.append(messages)

    inputs = tokenizer.apply_chat_template(
        batch_messages, tokenize=True, add_generation_prompt=True, 
        return_tensors="pt", return_dict=True, 
        padding=True,  # Enable padding
        truncation=True,  # Enable truncation
    )
    return inputs


def main_generate(args):
    dataset = args.dataset
    split = args.split
    model_name = args.model_name
    # model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    # model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    # model_name = 'Qwen/Qwen2.5-7B-Instruct'

    ### load LM and tokenizer ###
    device = torch.device('cuda')
    if 'Llama' in model_name:
        full_model_name = f'meta-llama/{model_name}'
    elif 'Qwen' in model_name:
        full_model_name = f'Qwen/{model_name}'
    elif 'Mistral' in model_name:
        full_model_name = f'mistralai/{model_name}'

    model = AutoModelForCausalLM.from_pretrained(
        full_model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #############################

    ### load QA dataset ###
    qa_ds = load_qa_ds(dataset, split)
    
    #############################
    out_root_dir = f"{args.out_root_dir}/{args.dataset}/{args.prompt_type}/{model_name}/"
    results_fn = f"{split}_{args.temperature}.jsonl"
    os.makedirs(out_root_dir, exist_ok=True)
    if os.path.exists(join(out_root_dir, results_fn)):
        with jsonlines.open(join(out_root_dir, results_fn), 'r') as f:
            history = list(f)
    else:
        with jsonlines.open(join(out_root_dir, results_fn), 'w') as f:
            history = []
    history_i = len(history)
    print(f"History exists. Start from {history_i}")

    ### generate answers ###
    if args.temperature == 0.1:
        n_response_per_question = 1
        batch_size = 64
    else:
        n_response_per_question = 10
        batch_size = 8

    all_example, all_local_prompts = [], []
    for i, x in enumerate(qa_ds):
        if i < history_i:
            continue
        question = x['question']
        local_prompt = utils.make_prompt(args.prompt_type, question)
        all_local_prompts.append(local_prompt)
        all_example.append(x)

    for it in tqdm(range(0, len(all_example), batch_size), total=len(all_example)//batch_size):
        if (it + 1 % 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()

        batch_local_prompt = all_local_prompts[it:it+batch_size]
        batch_example = all_example[it:it+batch_size]
        inputs = prepare_inputs(tokenizer, batch_local_prompt)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=args.temperature,
                num_return_sequences=n_response_per_question
            )
            answer_tokens = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True) # (n_res_per_q)
            answer_tokens = [ans.strip() for ans in answer_tokens]

        assert len(batch_example) * n_response_per_question == len(answer_tokens)
        for j, x in  enumerate(batch_example):
            x['model answers'] = answer_tokens[n_response_per_question*j:n_response_per_question*(j+1)]
            with jsonlines.open(join(out_root_dir, results_fn), 'a') as f:
                f.write(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_root_dir', default='/home/ziweiji/Hallu_Det/sem_uncertainty/outputs/', type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--prompt_type', default='sentence', type=str)
    parser.add_argument('--max_new_tokens', default=100, type=int)
    parser.add_argument('--dataset', default='pop_qa', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--model_name', default='Meta-Llama-3.1-8B-Instruct', type=str)
    parser.add_argument('--entailment_model', default='', type=str)
    
    args = parser.parse_args()
    main_generate(args)
    