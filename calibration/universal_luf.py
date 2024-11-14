import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import argparse
import sys
import os
current_path = os.getcwd()
root_path = os.path.abspath(os.path.join(current_path, os.pardir))
sys.path.append(root_path)
from verbal_uncertainty.prompts import get_qa_system_prompt
from sem_uncertainty.semantic_entropy.utils.utils import make_prompt
import json


def main(args):
    global model, tokenizer, device
    global results_df, ling_uncertain_idx, ling_certain_idx

    dataset = args.dataset
    model_name = args.model_name
    output_dir = f'./outputs/{dataset}/{model_name}/{args.prompt_type}/{split}'
    os.makedirs(output_dir, exist_ok=True)

    if args.prompt_type == 'uncertainty':
        sys_prompt = get_qa_system_prompt('uncertainty')
    
    Hs_questions_uncertain_ling = []
    Hs_questions_certain_ling = []

    for i in tqdm(ling_uncertain_idx):
        question = results_df['question'][i]
        if args.prompt_type == 'uncertainty':
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Question: {question}\nAnswer: "},
            ]
        elif args.prompt_type == 'sentence':
            local_prompt = make_prompt('sentence', question)
            messages = [
                {"role": "user", "content": local_prompt},
            ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, 
            return_tensors="pt", return_dict=True
        )

        with torch.no_grad():
            Hs_i = model(**inputs.to(device), output_hidden_states=True).hidden_states
            Hs_i = torch.cat(Hs_i, dim=0)[1:,-1].cpu()
            Hs_questions_uncertain_ling.append(Hs_i)


    for i in tqdm(ling_certain_idx):
        question = results_df['question'][i]
        if args.prompt_type == 'uncertainty':
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Question: {question}\nAnswer: "},
            ]
        elif args.prompt_type == 'sentence':
            local_prompt = make_prompt('sentence', question)
            messages = [
                {"role": "user", "content": local_prompt},
            ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, 
            return_tensors="pt", return_dict=True
        )

        with torch.no_grad():
            Hs_i = model(**inputs.to(device), output_hidden_states=True).hidden_states
            Hs_i = torch.cat(Hs_i, dim=0)[1:,-1].cpu()
            Hs_questions_certain_ling.append(Hs_i)

    Hs_questions_uncertain_ling = torch.stack(Hs_questions_uncertain_ling)
    Hs_questions_certain_ling = torch.stack(Hs_questions_certain_ling)
    Hs_hedge_kuq = Hs_questions_uncertain_ling.mean(0) - Hs_questions_certain_ling.mean(0)
    
    torch.save(Hs_questions_uncertain_ling, f'{output_dir}/uncertain_ling.pt')
    torch.save(Hs_questions_certain_ling, f'{output_dir}/certain_ling.pt')
    torch.save(Hs_hedge_kuq, f'{output_dir}/Hs_hedge_universal.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--prompt_type", type=str)
    args = parser.parse_args()


    # model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    model_name = args.model_name
    if 'Llama' in model_name:
        full_model_name = f'meta-llama/{model_name}'
    elif 'Qwen' in model_name:
        full_model_name = f'Qwen/{model_name}'
    elif 'Mistral' in model_name:
        full_model_name = f'mistralai/{model_name}'
    elif 'llama-3.1-8B-grpo' in model_name:
        full_model_name = f'AlistairPullen/{model_name}'
    ### load `LM and tokenizer ###
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(
        full_model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ###################`##########


    dataset = args.dataset
    split = args.split
    # if model_name == 'Meta-Llama-3.1-8B-Instruct':
    #     results_df = pd.read_csv(f"{root_path}/datasets/{dataset}/sampled/{split}.csv")
    # else:
    results_df = pd.read_csv(f"{root_path}/datasets/{dataset}/{model_name}/{split}.csv")
    # data = pd.read_csv(f"/home/ziweiji/Hallu_Det/datasets/{dataset}/sentence/{split}.csv")
    # assert len(results_df) == len(data)
    # results_df['model_generated'] = data['model_generated']
    lu_scores_llm = results_df['ling_uncertainty'].to_numpy()
    answerable_labels = len(lu_scores_llm) * [1]
    

    # use threshold
    ling_uncertain_idx = np.array([
        i for i in range(len(results_df))
        if answerable_labels[i] == 1 and lu_scores_llm[i] >= 0.9
    ])
    ling_certain_idx = np.array([
        i for i in range(len(results_df))
        if answerable_labels[i] == 1 and lu_scores_llm[i] <= 0.05
    ])
    
    print(len(ling_uncertain_idx), len(ling_certain_idx))


    main(args)