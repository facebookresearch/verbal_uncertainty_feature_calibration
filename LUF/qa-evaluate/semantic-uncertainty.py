import argparse
import csv
import os
import pickle
import random
import json
import itertools
from os.path import join

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from utils import get_qa_system_prompt

evaled_model_names = [
    'meta-llama/Meta-Llama-3.1-8B-Instruct', 
    # 'meta-llama/Meta-Llama-3.1-70B-Instruct'
]

dataset_names = [
    'nq_open', 'trivia_qa', 'pop_qa'
]
n_chunk = 10
chunk_idx = list(range(n_chunk))
slurm_arr_args = list(itertools.product(evaled_model_names, dataset_names, chunk_idx))



def prepare_inputs(tokenizer, question, model_answers, i):
    input_texts = []
    qa_0 = f"Question: {question} Answer: {model_answers[i]}"
    for j in range(len(model_answers)):
        qa_1 = f"Question: {question} Answer: {model_answers[j]}"
        input_text_ij = qa_0 + ' [SEP] ' + qa_1
        input_texts.append(input_text_ij)
    inputs = tokenizer(input_texts, padding=True, return_tensors='pt')  # (n_ans, seq_len)
    return inputs


def semantic_clustering(nli_labels):
    # nli_labels: (n_ans, n_ans)
    sem_clusters = [[0]]
    for i in range(1, len(nli_labels)):
        n_belonged_cluster_i = 0
        for sem_cluster in sem_clusters:
            n_consistent = 0
            for j in sem_cluster:
                label_ij = nli_labels[i][j]
                label_ji = nli_labels[j][i]
                if label_ij + label_ji > 2:
                    n_consistent += 1
            if n_consistent == len(sem_cluster):
                sem_cluster.append(i)
                n_belonged_cluster_i += 1
        if n_belonged_cluster_i == 0:
            sem_clusters.append([i])
    return sem_clusters


def semantic_entropy(nli_labels, log_likelihoods):
    sem_clusters = semantic_clustering(nli_labels)
    cluster_log_likelihoods = []
    for sem_cluster in sem_clusters:
        cluster_log_likelihood = torch.logsumexp(log_likelihoods[sem_cluster], dim=0)
        cluster_log_likelihoods.append(cluster_log_likelihood)
    cluster_log_likelihoods = torch.tensor(cluster_log_likelihoods)
    sem_entropy = -torch.sum(cluster_log_likelihoods) / cluster_log_likelihoods.shape[0]

    return sem_entropy.item()


def get_ans_log_likelihoods(model, tokenizer, device, result, system_prompt):
    question, answers = result['question'], result['model answers']
    formatted_input_text_q = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\nAnswer: "},
        ], tokenize=False, add_generation_prompt=True 
    )
    n_tok_q = len(tokenizer(formatted_input_text_q).input_ids)
    
    log_likelihoods_ans = []
    for ans in answers:
        inputs = tokenizer(
            formatted_input_text_q + ans, 
            return_tensors='pt', padding=True
        )
        labels = torch.clone(inputs['input_ids'])
        labels[:, :n_tok_q] = -100
        inputs['labels'] = labels

        with torch.no_grad():
            loss = model(**inputs.to(device)).loss
            log_likelihood_ans_i = -loss.cpu()
            log_likelihoods_ans.append(log_likelihood_ans_i)

    return torch.stack(log_likelihoods_ans)



def main_nli(args):
    ##### load model generated answers #####
    evaled_model_name, ds_name, chunk_id = slurm_arr_args[args.job_id]
    evaled_model_name_short = evaled_model_name.split('/')[-1]
    results_fn = f"{evaled_model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}_ling.json"
    with open(join(args.results_dir, results_fn), 'r') as f:
        results = json.load(f)
    ##########################################

    ##### load deberta nli model #####
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-large-mnli"
    ).to(device)  
    # "id2label": {
    #     "0": "CONTRADICTION",
    #     "1": "NEUTRAL",
    #     "2": "ENTAILMENT"
    #   }
    ##########################################

    ##### take first 100 result examples #####
    # results = results[:10]
    ##########################################

    ##### get nli labels between each answer pair #####
    for result in tqdm(results):
        nli_labels = []
        question, model_answers = result['question'], result['model answers']
        for i in range(len(model_answers)):
            inputs = prepare_inputs(tokenizer, question, model_answers, i).to(device)
            with torch.no_grad():
                outputs = nli_model(**inputs)['logits']
                nli_labels_i = torch.argmax(outputs, dim=1).cpu().tolist()
                nli_labels.append(nli_labels_i)

        result['nli labels'] = nli_labels
    ##########################################

    ### save nli results ###
    results_fn = f"{evaled_model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}_ling_sem.json"
    with open(join(args.results_dir, results_fn), 'w') as f:
        json.dump(results, f)
    #############################



def main_entropy(args):
    ##### load model generated answers #####
    evaled_model_name, ds_name, chunk_id = slurm_arr_args[args.job_id]
    evaled_model_name_short = evaled_model_name.split('/')[-1]
    results_fn = f"{evaled_model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}_ling_sem.json"
    with open(join(args.results_dir, results_fn), 'r') as f:
        results = json.load(f)
    ##########################################

    ### load LM and tokenizer ###
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(
        evaled_model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval();
    tokenizer = AutoTokenizer.from_pretrained(evaled_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #############################

    ### get system prompt ###
    sys_prompt = get_qa_system_prompt(args.prompt_method)
    #############################

    ### get answer log likelihoods and semantic entropies ###
    for result in tqdm(results):
        log_likelihoods = get_ans_log_likelihoods(model, tokenizer, device, result, sys_prompt)
        sem_entropy = semantic_entropy(result['nli labels'], log_likelihoods)
        result['semantic entropy'] = sem_entropy
        result['answer log likelihoods'] = log_likelihoods.tolist()
    #############################
    
    ### save results ###
    results_fn = f"{evaled_model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}_ling_sem.json"
    with open(join(args.results_dir, results_fn), 'w') as f:
        json.dump(results, f)
    #############################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', default='/data/home/jadeleiyu/hall-irrelevant-context/data', type=str)
    parser.add_argument('--results_dir', default='/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/qa-eval-results/', type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--n_response_per_question', default=20, type=int)
    parser.add_argument('--job_id', default=0, type=int)
    parser.add_argument('--prompt_method', default='uncertainty', type=str)
    
    args = parser.parse_args()
    # main_nli(args)
    main_entropy(args)
