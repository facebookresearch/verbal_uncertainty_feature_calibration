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
import json

import os
os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'

import sys
sys.path.append('/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/LUF/')
from utils import DECISIVENESS_SYS_PROMPT


evaled_model_names = [
    'meta-llama/Meta-Llama-3.1-8B-Instruct', 
    # 'meta-llama/Meta-Llama-3.1-70B-Instruct'
]
judge_model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
judge_model_name_short = judge_model_name.split('/')[1]

dataset_names = [
    'nq_open', 'trivia_qa', 'pop_qa',
    'SelfAware', 'KUQ'
]

n_chunk = 10
chunk_idx = list(range(n_chunk))
slurm_arr_args = list(itertools.product(evaled_model_names, dataset_names, chunk_idx))

def prepare_inputs(tokenizer, example):
    input_texts = []
    question = example['question']
    for answer in example['model answers']:
        messages = [
            {"role": "system", "content": DECISIVENESS_SYS_PROMPT},
            {"role": "user", "content": f"Question: {question} Proposed answer: {answer}"},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_texts.append(input_text)
    inputs = tokenizer(
        input_texts, 
        return_tensors='pt',
        padding='longest'
    )
    
    return inputs


def extract_judge_results(judge_output_text):
    try:
        decisiveness_score = float(judge_output_text.split('Decisiveness score: ')[1].strip())
        ling_uncertain_score = 1. - decisiveness_score
    except Exception as e:
        ling_uncertain_score = -1.
    
    return ling_uncertain_score


def main(args):
    ###### load llama-3.1-70B as the judge ######
    device = torch.device('cuda')
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_model_name, torch_dtype=torch.float16, 
        device_map='auto'
    )
    judge_model.eval();
    tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    ##########################################

    ##### load model generated answers #####
    evaled_model_name, ds_name, chunk_id = slurm_arr_args[args.job_id]
    evaled_model_name_short = evaled_model_name.split('/')[-1]
    qa_ds_fn = f"{evaled_model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}.json"
    with open(join(args.results_dir, qa_ds_fn), 'r') as f:
        qa_ds = json.load(f)
    ##########################################

    ##### take first 100 result examples #####
    # qa_ds = qa_ds[:10]
    ##########################################

    ##### get judged decisiveness scores and extracted assertions #####
    ling_uncertain_scores = []
    for example in tqdm(qa_ds):
        inputs = prepare_inputs(tokenizer, example).to(device)
        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs, 
                max_new_tokens=10,
                do_sample=False
            )
            answer_tokens = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True) # (n_res_per_q)
            answer_tokens = [ans.strip() for ans in answer_tokens]
        ling_uncertain_scores_i = []
        for judge_output_text in answer_tokens:
            ling_uncertain_scores_i.append(
                extract_judge_results(judge_output_text)
            )
        # result['linguistic uncertainty score'] = ling_uncertain_scores_i
        ling_uncertain_scores.append(ling_uncertain_scores_i)
    ##########################################

    ### save judge results ###
    results_fn = f"{evaled_model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}_lu-llm-judge.p"
    with open(join(args.results_dir, results_fn), 'wb') as f:
        pickle.dump(ling_uncertain_scores, f)
    #############################


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/LUF/qa-evaluate/qa-eval-results', type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--n_response_per_question', default=20, type=int)
    parser.add_argument('--job_id', default=0, type=int)
    parser.add_argument('--prompt_method', default='uncertainty', type=str)
    
    args = parser.parse_args()
    main(args)

