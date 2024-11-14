import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import pandas as pd
from tqdm.auto import tqdm
from os.path import join
import numpy as np
from ast import literal_eval
import argparse
import itertools
import pickle
from datasets import load_dataset

from iti_utils import remove_all_hooks, register_feature_ablation_hook, eval_ans_hedgeness_emb

import sys
sys.path.append('/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/LUF/')
from utils import get_qa_system_prompt

import os
import json
os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'

results_dir = '/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/LUF/qa-evaluate/qa-eval-results'
model_names = [
    'meta-llama/Meta-Llama-3.1-8B-Instruct', 
    # 'meta-llama/Meta-Llama-3.1-70B-Instruct'
]
dataset_names = [
    'nq_open', 'trivia_qa', 'pop_qa'
]
iti_alphas = np.arange(0.1, 1.1, 0.1)

slurm_arr_args = list(itertools.product(model_names, dataset_names, iti_alphas))


def load_qa_results(
    ds_name, results_dir, model_name_short, 
    prompt_method='uncertainty', 
    temperature=1.0, n_chunk=10
    ):

    results_df = {
        'question': [],
        'lu-llm-judge': [],
        'lu-emb-esu': [],
        'lu-emb-euu': [],
    }
    chunk_idx = np.arange(n_chunk)

    for chunk_id in tqdm(chunk_idx):
        qa_results_fn = join(results_dir, f"{model_name_short}_{ds_name}_{chunk_id}_{prompt_method}_{temperature}.json")
        lu_llm_judge_fn = join(results_dir, f"{model_name_short}_{ds_name}_{chunk_id}_{prompt_method}_{temperature}_lu-llm-judge.p")
        lu_emb_esu_fn = join(results_dir, f"{model_name_short}_{ds_name}_{chunk_id}_{prompt_method}_{temperature}_lu-emb-sim-esu.p")
        lu_emb_euu_fn =join(results_dir, f"{model_name_short}_{ds_name}_{chunk_id}_{prompt_method}_{temperature}_lu-emb-sim-euu.p")
        if os.path.isfile(lu_llm_judge_fn) and os.path.isfile(lu_emb_esu_fn) and os.path.isfile(lu_emb_euu_fn):
            with open(join(results_dir, qa_results_fn), 'r') as f:
                qa_ds = json.load(f)
            with open(join(results_dir, lu_llm_judge_fn), 'rb') as f:
                lu_llm_judge = pickle.load(f)
            with open(join(results_dir, lu_emb_esu_fn), 'rb') as f:
                lu_emb_esu = pickle.load(f)
            with open(join(results_dir, lu_emb_euu_fn), 'rb') as f:
                lu_emb_euu = pickle.load(f)

            for i in range(len(qa_ds)):
                question = qa_ds[i]['question']
                lu_llm_judge_i = np.array(lu_llm_judge[i]).mean()
                lu_emb_esu_i = lu_emb_esu[i]
                lu_emb_euu_i = lu_emb_euu[i]

                results_df['question'].append(question)
                results_df['lu-llm-judge'].append(lu_llm_judge_i)
                results_df['lu-emb-esu'].append(lu_emb_esu_i)
                results_df['lu-emb-euu'].append(lu_emb_euu_i)
    
    return pd.DataFrame(results_df)



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
    model_name, ds_name, iti_alpha = slurm_arr_args[args.job_id]
    model_name_short = model_name.split('/')[-1]

    print(f'model_name_short: {model_name_short}')
    print(f'ds_name: {ds_name}')
    print(f'iti_alpha: {iti_alpha}')
    print()
    
    ### load LM and tokenizer ###
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #############################

    ##### load lu expression embedding model (a fine-tuned Mistral-7B by Nvidia) #####
    emb_model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True).to('cuda')
    ##########################################

    ### load QA results ###
    qa_df = load_qa_results(ds_name, results_dir, model_name_short)
    #############################

    ### randomly sample 1000 examples of high and low ling uncertainty ###
    sorted_qa_df = qa_df.sort_values(by=['lu-llm-judge'])
    qa_df_certain = sorted_qa_df.head(1000).reset_index()
    qa_df_uncertain = sorted_qa_df.tail(1000).reset_index()
    # qa_df = qa_df.sample(1000).reset_index()
    # print(f'number of high certain examples: {qa_df_certain.shape[0]}')
    # print(f'number of high uncertain examples: {qa_df_uncertain.shape[0]}')
    #############################

    ### get system prompt ###
    sys_prompt = get_qa_system_prompt(args.prompt_method)
    #############################

    ### generate answers ###
    remove_all_hooks(model)
    torch.cuda.empty_cache()

    ### load universal hedgeness features and register iti hooks ###
    ling_uncertain_feats = torch.load(
        '/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/LUF/rep-analysis/outputs/Hs_hedge_KUQ_universal.pt'
    )
    layer_idx = np.arange(32)
    #############################

    ### generate answers for low hedgeness questions with hedgeness feature restoration ###
    register_feature_ablation_hook(model, ling_uncertain_feats, layer_idx, iti_alpha)

    iti_results_df_certain = {
        'question': [],
        'model answer with iti': [],
        'hedgeness score before iti': [],
    }
    for i in tqdm(range(qa_df_certain.shape[0])):
        question = qa_df_certain['question'][i]
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
        iti_results_df_certain['question'].append(question)
        iti_results_df_certain['model answer with iti'].append(answer_tokens)
        iti_results_df_certain['hedgeness score before iti'].append(qa_df_certain['lu-llm-judge'][i])
    
    remove_all_hooks(model)
    torch.cuda.empty_cache()

    iti_results_df_certain = pd.DataFrame(iti_results_df_certain)

    ### evaluate answer hedgeness using esu ###
    lu_scores_emb_certain = eval_ans_hedgeness_emb(emb_model, iti_results_df_certain)
    iti_results_df_certain['hedgeness score after iti'] = lu_scores_emb_certain

    ### save results ###
    iti_results_df_certain.to_csv(f'outputs/iti_results_{model_name_short}_{ds_name}_{iti_alpha}_hedge_restore.csv')
    #############################

    ### generate answers for high hedgeness questions with hedgeness feature ablation ###
    register_feature_ablation_hook(model, ling_uncertain_feats, layer_idx, -iti_alpha)

    iti_results_df_uncertain = {
        'question': [],
        'model answer with iti': [],
        'hedgeness score before iti': [],
    }
    for i in tqdm(range(qa_df_uncertain.shape[0])):
        question = qa_df_uncertain['question'][i]
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
        iti_results_df_uncertain['question'].append(question)
        iti_results_df_uncertain['model answer with iti'].append(answer_tokens)
        iti_results_df_uncertain['hedgeness score before iti'].append(qa_df_uncertain['lu-llm-judge'][i])
    
    remove_all_hooks(model)
    torch.cuda.empty_cache()
    #############################

    iti_results_df_uncertain = pd.DataFrame(iti_results_df_uncertain)

    ### evaluate answer hedgeness using esu ###
    lu_scores_emb = eval_ans_hedgeness_emb(emb_model, iti_results_df_uncertain)
    iti_results_df_uncertain['hedgeness score after iti'] = lu_scores_emb
    #############################

    ### save results ###
    iti_results_df_uncertain.to_csv(f'outputs/iti_results_{model_name_short}_{ds_name}_{iti_alpha}_hedge_ablate.csv')
    #############################

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--n_response_per_question', default=20, type=int)
    parser.add_argument('--job_id', default=0, type=int)
    parser.add_argument('--prompt_method', default='uncertainty', type=str)
    
    args = parser.parse_args()
    main_generate(args)




