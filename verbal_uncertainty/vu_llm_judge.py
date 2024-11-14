# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
set_seed(42)
from tqdm.auto import tqdm
from os.path import join
import argparse
import json
import jsonlines
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
sys.path.append(current_dir)
from prompts import DECISIVENESS_SYS_PROMPT
from tqdm.contrib.concurrent import thread_map
import openai

def prepare_inputs(example):
    input_texts = []
    question = example['question']
    if 'responses' in example:
        responses = example['responses']
    else:
        responses = example['model answers']
    for answer in responses:
        messages = [
            {"role": "system", "content": DECISIVENESS_SYS_PROMPT},
            {"role": "user", "content": f"Question: {question} Proposed answer: {answer}"},
        ]
        input_texts.append(messages)
    return input_texts, len(responses)


def extract_judge_results(judge_output_text):
    try:
        decisiveness_score = float(judge_output_text.split('Decisiveness score: ')[1].strip())
        verbal_uncertain_score = 1. - decisiveness_score
    except Exception as e:
        verbal_uncertain_score = -1.
    
    return verbal_uncertain_score


def get_batch_results(judge_model, tokenizer, batch_message):
    if type(judge_model) == str:
        def predict(messages, temperature):
            assert type(messages) == list
            client = openai.OpenAI(
                base_url=judge_model,
                api_key="NOT A REAL KEY",
            )
            chat_completion = client.chat.completions.create(
                model='meta-llama/Llama-3.1-70B-Instruct',
                messages=messages,
                max_tokens=10,
                temperature=temperature,
            )
            return chat_completion.choices[0].message.content
        answer_tokens = thread_map(
            lambda p: predict(p, temperature=0.9),
            batch_message,
            max_workers=40,
            desc="using vllm")
    else:
        device = judge_model.device
        inputs = tokenizer.apply_chat_template(
                batch_message, tokenize=True, add_generation_prompt=True, 
                truncation=True, padding=True,
                return_tensors="pt", return_dict=True).to(device)

        with torch.no_grad():
            outputs = judge_model.generate(
                **inputs, 
                max_new_tokens=10,
                do_sample=False
            )
            answer_tokens = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True) # (n_res_per_q)
    
    answer_tokens = [ans.strip() for ans in answer_tokens]
    verbal_uncertain_scores_batch = []
    for judge_output_text in answer_tokens:
        verbal_uncertain_scores_batch.append(
            extract_judge_results(judge_output_text)
        )
    return verbal_uncertain_scores_batch

def judge_main(args):
    dataset = args.dataset
    split = args.split
    batch_size = args.batch_size
    ##### load model generated answers #####
    evaled_model_name = args.model_name

    if args.input_file:
        input_file = args.input_file
    else:
        input_file = f"{evaled_model_name}_{dataset}_{split}_{args.prompt_method}_{args.temperature}.jsonl"

    out_root_dir = f"{current_dir}/{args.results_dir}"
    with jsonlines.open(join(out_root_dir, input_file), 'r') as f:
        qa_ds = list(f)
    ##########################################
    if args.results_fn:
        results_fn = args.results_fn
    else:
        results_fn = f"{evaled_model_name}_{dataset}_{split}_{args.prompt_method}_{args.temperature}_vu-llm-judge.json"
    verbal_uncertain_scores = []
    if os.path.exists(join(out_root_dir, results_fn)):
        with open(join(out_root_dir, results_fn), 'r') as f:
            verbal_uncertain_scores = json.load(f)
    history_i = len(verbal_uncertain_scores)

    ##### get judged decisiveness scores and extracted assertions #####
    all_message = []
    for i, example in tqdm(enumerate(qa_ds), total=len(qa_ds)):
        if i < history_i:
            continue
        input_texts, N = prepare_inputs(example)
        all_message.extend(input_texts)
    print('len(all_message)', len(all_message))
    print('history_i', history_i)
    print('N', N)
    if len(all_message) == 0:
        return
    
    ###### load llama-3.1-70B as the judge ######
    if args.port: # VLM
        judge_model = args.port
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
    
    for i in tqdm(range(0, len(all_message), batch_size)):
        batch_message = all_message[i:i+batch_size]
        verbal_uncertain_scores_batch = get_batch_results(judge_model, tokenizer, batch_message)
        
        # group verbal_uncertain_scores_batch each N
        verbal_uncertain_scores_q = [] # for each question
        assert len(verbal_uncertain_scores_batch) % N == 0
        for j in range(len(verbal_uncertain_scores_batch)//N):
            verbal_uncertain_scores_q.append(
                verbal_uncertain_scores_batch[j*N:(j+1)*N]
            )
        verbal_uncertain_scores.extend(verbal_uncertain_scores_q)

        if i%(batch_size*5) == 0: # save every 10 batch
            with open(join(out_root_dir, results_fn), 'w') as f:
                assert len(verbal_uncertain_scores)
                json.dump(verbal_uncertain_scores, f)

    ### save judge results ###
    with open(join(out_root_dir, results_fn), 'w') as f:
        assert len(verbal_uncertain_scores)
        json.dump(verbal_uncertain_scores, f)
    #############################


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='outputs', type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--input_file', default=None, type=str)
    parser.add_argument('--results_fn', default=None, type=str)
    parser.add_argument('--prompt_method', default='uncertainty', type=str)
    parser.add_argument('--dataset', default='pop_qa', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--port', type=str, default="")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=80)
    args = parser.parse_args()
    judge_main(args)

