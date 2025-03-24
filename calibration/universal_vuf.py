import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
sys.path.append(f"{root_path}/verbal_uncertainty")
from prompts import get_qa_system_prompt
sys.path.append(f"{root_path}/sem_uncertainty/")
from semantic_entropy.utils import make_prompt

def main(args):
    global model, tokenizer, device
    global results_df, verbal_uncertain_idx, verbal_certain_idx

    dataset = args.dataset
    model_name = args.model_name
    output_dir = f'{current_dir}/outputs/{dataset}/{model_name}/{args.prompt_type}/{split}'
    os.makedirs(output_dir, exist_ok=True)

    if args.prompt_type == 'uncertainty':
        sys_prompt = get_qa_system_prompt('uncertainty')
    
    Hs_questions_uncertain_verbal = []
    Hs_questions_certain_verbal = []

    for i in tqdm(verbal_uncertain_idx):
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
            Hs_questions_uncertain_verbal.append(Hs_i)


    for i in tqdm(verbal_certain_idx):
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
            Hs_questions_certain_verbal.append(Hs_i)

    Hs_questions_uncertain_verbal = torch.stack(Hs_questions_uncertain_verbal)
    Hs_questions_certain_verbal = torch.stack(Hs_questions_certain_verbal)
    Hs_hedge_kuq = Hs_questions_uncertain_verbal.mean(0) - Hs_questions_certain_verbal.mean(0)
    
    torch.save(Hs_questions_uncertain_verbal, f'{output_dir}/uncertain_verbal.pt')
    torch.save(Hs_questions_certain_verbal, f'{output_dir}/certain_verbal.pt')
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
    elif 'Llama-3.1-8B-GRPO-Instruct' in model_name:
        full_model_name = f'ymcki/{model_name}'
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
    results_df = pd.read_csv(f"{root_path}/datasets/{dataset}/{model_name}/{split}.csv")
    # assert len(results_df) == len(data)
    # results_df['model_generated'] = data['model_generated']
    verbal_scores_llm = results_df['verbal_uncertainty'].to_numpy()
    answerable_labels = len(verbal_scores_llm) * [1]
    

    # use threshold
    verbal_uncertain_idx = np.array([
        i for i in range(len(results_df))
        if answerable_labels[i] == 1 and verbal_scores_llm[i] >= 0.9
    ])
    verbal_certain_idx = np.array([
        i for i in range(len(results_df))
        if answerable_labels[i] == 1 and verbal_scores_llm[i] <= 0.05
    ])
    
    print(len(verbal_uncertain_idx), len(verbal_certain_idx))

    main(args)