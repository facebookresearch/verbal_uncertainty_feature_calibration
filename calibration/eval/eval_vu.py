# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
set_seed(42)
import pandas as pd
from tqdm.auto import tqdm
import argparse
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_path)
from verbal_uncertainty.vu_llm_judge import prepare_inputs, get_batch_results
from src.eval_utils import get_available_servers

def judge_main(args):
    batch_size = args.batch_size
    max_alpha = args.max_alpha
    model_name = args.model_name
    prompt_type = args.prompt_type
    dataset = args.dataset
    split = args.split
    iti_method = args.iti_method
    str_process_layers = args.str_process_layers

    ##### load model generated answers #####
    if args.use_predicted:
        output_base_dir = f"{root_path}/calibration/predicted_outputs/{dataset}/{model_name}/{prompt_type}/{split}"
    else:
        if iti_method == 5:
            output_base_dir = f"{root_path}/calibration/outputs2/{dataset}/{model_name}/{prompt_type}/{split}"
        else:
            output_base_dir = f"{root_path}/calibration/outputs/{dataset}/{model_name}/{prompt_type}/{split}"
    
    if args.input_file:
        input_path = f'{output_base_dir}/{input_path}'
    else:
        if prompt_type == 'uncertainty': # semantic control
            if iti_method in [0, 2, 3, 4]:
                input_path = f'{output_base_dir}/with_vufi_{iti_method}_{str_process_layers}_{max_alpha}.jsonl'
            elif iti_method == 1:
                input_path = f'{output_base_dir}/with_vufi_{iti_method}_trivia_qa_{str_process_layers}_{max_alpha}.jsonl'
        else: # causal
            if iti_method in [0, 2, 3, 5]:
                input_path = f'{output_base_dir}/questions_{args.question_type}_with_vufi_{iti_method}_{str_process_layers}_{max_alpha}.jsonl'
            elif iti_method == 1:
                input_path = f'{output_base_dir}/questions_{args.question_type}_with_vufi_{iti_method}_{args.dataset2}_{str_process_layers}_{max_alpha}.jsonl'
    
    results_fn = input_path.replace("with_vufi", 'vu')[:-1]
    print('input_path', input_path)
    qa_ds = pd.read_json(input_path, lines=True)
    print("len(qa_ds)", len(qa_ds))
    ##########################################
    verbal_uncertain_scores = {} # question: score
    if os.path.exists(results_fn):
        try:
            with open(results_fn, 'r') as f:
                verbal_uncertain_scores = json.load(f)
        except Exception as e:
            print('Error loading', results_fn, e)
    
    ##### get judged decisiveness scores and extracted assertions #####
    all_message = []
    all_question = []
    for i, example in qa_ds.iterrows():
        if not example["responses"]:
            question = example['question']
            verbal_uncertain_scores[question] = [-1]
            continue
        if example['question'] in verbal_uncertain_scores:
            continue
        input_texts, N = prepare_inputs(example)
        all_message.extend(input_texts)
        all_question.extend([example['question']]*N)
        
    print('len(all_message)', len(all_message))

    if all_message:
        print('N', N)
        batch_size = batch_size * N
        print('batch_size', batch_size)
        ###### load llama-3.1-70B as the judge ######
        if args.entailment_model.startswith("http"): # VLM
            judge_model = args.entailment_model
            tokenizer = None
        else:
            judge_model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
            judge_model = AutoModelForCausalLM.from_pretrained(
                judge_model_name, torch_dtype=torch.float16, 
                device_map='auto'
            )
            judge_model.eval()
            tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        ##########################################
    
    for i in tqdm(range(0, len(all_message), batch_size), total=len(all_message)//batch_size):
        batch_message = all_message[i:i+batch_size]
        batch_question = all_question[i:i+batch_size]

        verbal_uncertain_scores_batch = get_batch_results(judge_model, tokenizer, batch_message)
        assert len(verbal_uncertain_scores_batch) == len(batch_message)
        # group verbal_uncertain_scores_batch each N
        
        # for each question
        for j in range(len(verbal_uncertain_scores_batch)//N):
            question = batch_question[j*N]
            assert question == batch_question[j*N+1] == batch_question[j*N+2]
            verbal_uncertain_scores[question] = verbal_uncertain_scores_batch[j*N:(j+1)*N]
            
        with open(results_fn, 'w') as f:
            print('len(verbal_uncertain_scores)', len(verbal_uncertain_scores))
            json.dump(verbal_uncertain_scores, f)

    ### save judge results ###
    with open(results_fn, 'w') as f:
        print('len(verbal_uncertain_scores)', len(verbal_uncertain_scores))
        json.dump(verbal_uncertain_scores, f)
    #############################
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--dataset2', default=None, type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--prompt_type', default=None, type=str)
    parser.add_argument('--input_file', default=None, type=str)
    parser.add_argument('--split', default=None, type=str)
    parser.add_argument('--question_type', default=None, type=str)
    parser.add_argument('--max_alpha', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--entailment_model', type=str, default="")
    parser.add_argument("--use_predicted", type=int, default=0)
    parser.add_argument("--iti_method", type=int, default=2)
    parser.add_argument("--str_process_layers", type=str, default="")
    args = parser.parse_args()
    
    server_dict = get_available_servers()["meta-llama/Llama-3.1-70B-Instruct"]
    server_urls = server_dict["server_urls"]
    if "http" not in args.entailment_model:
        args.entailment_model = server_urls[int(args.entailment_model)]

    judge_main(args)