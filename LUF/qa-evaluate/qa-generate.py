import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm.auto import tqdm
from os.path import join
import numpy as np
from ast import literal_eval
import argparse
import itertools
import pickle
from datasets import load_dataset

import sys
sys.path.append('/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/LUF/')
from utils import get_qa_system_prompt

import os
import json
os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'


model_names = [
    # 'meta-llama/Meta-Llama-3.1-8B-Instruct', 
    'meta-llama/Meta-Llama-3.1-70B-Instruct'
]
dataset_names = [
    # 'nq_open', 
    # 'trivia_qa', 'pop_qa',
    # 'SelfAware', 'KUQ'
    'pop_qa', 'KUQ'
]
n_chunk = 10
chunk_idx = list(range(n_chunk))
slurm_arr_args = list(itertools.product(model_names, dataset_names, chunk_idx))

def load_qa_ds(dataset_name):
    qa_ds = []
    if dataset_name == 'nq_open':
        # nq_open_validation: 3610 questions
        hf_ds = load_dataset("google-research-datasets/nq_open")['validation']
        for x in hf_ds:
            qa_ds.append({
                'question': x['question'],
                'answer': x['answer'],
                'answerable': 1
            })

    elif dataset_name == 'trivia_qa':
        # trivia_qa_validation: 17944 questions
        hf_ds = load_dataset("mandarjoshi/trivia_qa", "rc")['validation']
        for x in hf_ds:
            qa_ds.append({
                'question': x['question'],
                'answer': x['answer']['aliases'],
                'answerable': 1
            })

    elif dataset_name == 'pop_qa':
        # pop_qa_test: 14267 questions
        hf_ds = load_dataset("akariasai/PopQA")['test']
        for x in hf_ds:
            qa_ds.append({
                'question': x['question'],
                'answer': eval(x['possible_answers']),
                'answerable': 1
            })

    elif dataset_name == 'SelfAware':
        # SelfAware: 3369 questions
        for split in ['train', 'dev', 'test']:
            with open(f"/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/data/{dataset_name}/{dataset_name}.ncan.{split}.json", 'r') as f:
                qa_data = json.load(f)
            for x in qa_data:
                qa_ds.append({
                'question': x[0]['content'],
                'answer': 'N/A',
                'answerable': int(eval(x[0]['answerable']))
            })
    
    elif dataset_name == 'KUQ':
        # KUQ: 4777 questions
        for split in ['train', 'dev', 'test']:
            with open(f"/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/data/{dataset_name}/{dataset_name}.ncan.{split}.json", 'r') as f:
                qa_data = json.load(f)
            for x in qa_data:
                qa_ds.append({
                'question': x[0]['content'],
                'answer': 'N/A',
                'answerable': 1 - int(eval(x[0]['unknown']))
            })
    
    return qa_ds



def prepare_inputs(tokenizer, question, sys_prompt):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Question: {question}\nAnswer: "},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, 
        return_tensors="pt", return_dict=True
    )
    return inputs


def main_generate(args):
    model_name, ds_name, chunk_id = slurm_arr_args[args.job_id]
    model_name_short = model_name.split('/')[-1]
    
    ### load LM and tokenizer ###
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #############################

    ### load QA dataset ###
    qa_ds = load_qa_ds(ds_name)
    #############################

    ### truncate ds by taking only the first 1000 examples ###
    # qa_ds = qa_ds[:100]
    #############################

    ### divide the ds into chunks ###
    data_idx_chunk = np.array_split(np.arange(len(qa_ds)), n_chunk)[chunk_id]
    qa_ds = [qa_ds[i] for i in data_idx_chunk]
    #############################

    ### get system prompt ###
    sys_prompt = get_qa_system_prompt(args.prompt_method)
    #############################

    ### generate answers ###
    model_answers = []
    for x in tqdm(qa_ds):   
        question = x['question']
        inputs = prepare_inputs(tokenizer, question, sys_prompt).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=args.n_response_per_question
            )
            answer_tokens = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True) # (n_res_per_q)
            answer_tokens = [ans.strip() for ans in answer_tokens]
        model_answers.append(answer_tokens)
    #############################

    ### save results ###
    for i in range(len(qa_ds)):
        qa_ds[i]['model answers'] = model_answers[i]
    results_fn = f"{model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}.json"
    with open(join(args.results_dir, results_fn), 'w') as f:
        json.dump(qa_ds, f)
    #############################


def main_merge(args):
    for model_name in model_names:
        model_name_short = model_name.split('/')[-1]
        for ds_name in dataset_names:
            merged_results = []
            for chunk_id in range(n_chunk):
                results_fn = f"{model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}.json"
                with open(join(args.results_dir, results_fn), 'r') as f:
                    chunk_results = json.load(f)
                merged_results += chunk_results
            
            merged_results_fn = f"{model_name_short}_{ds_name}_{args.prompt_method}_{args.temperature}.json"
            with open(merged_results_fn, 'w') as f:
                json.dump(merged_results, f)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/LUF/qa-evaluate/qa-eval-results', type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--n_response_per_question', default=5, type=int)
    parser.add_argument('--job_id', default=0, type=int)
    parser.add_argument('--prompt_method', default='uncertainty', type=str)
    
    args = parser.parse_args()
    main_generate(args)

    # main_merge()
