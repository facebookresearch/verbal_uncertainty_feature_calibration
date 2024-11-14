# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm.auto import tqdm
import argparse
import sys
sys.path.append(root_path)
from verbal_uncertainty.prompts import get_qa_system_prompt
from sem_uncertainty.semantic_entropy.utils import make_prompt
from src.utils import process_layers_to_process
# set seed 
torch.manual_seed(42)
np.random.seed(42)
import jsonlines
from collections import defaultdict
        
def register_feature_ablation_hook2(model, Hs_feature, process_layers, alpha):
    if not hasattr(model, '_ablation_hooks'):
        model._ablation_hooks = {}

    for l in process_layers:
        if l in model._ablation_hooks:
            model._ablation_hooks[l].remove()
            del model._ablation_hooks[l]

        device_idx_l = model.hf_device_map[f"model.layers.{l}"]
        h_feature_l = Hs_feature[l].to(f"cuda:{device_idx_l}")  # (h_dim)
        h_feature_l = h_feature_l / torch.sqrt(h_feature_l.pow(2).sum(-1))

        def make_feature_ablation_hook(h_feature_l, alpha):
            def feature_ablation_hook(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    outputs_0 = outputs[0]   # (B, seq_len, h_dim)
                    if outputs_0.shape[1] > 1:
                        # b = torch.matmul(outputs_0, h_feature_l).unsqueeze(-1)
                        outputs_0 += h_feature_l * alpha
                        # print('mean', b.mean().item(), 'std', b.std().item(), 'b', b)
                    return (outputs_0,) + outputs[1:]
                else:
                    assert False
                    if outputs.shape[1] > 1:
                        outputs += h_feature_l * alpha

                    return outputs
            return feature_ablation_hook

        handle = model.model.layers[l].register_forward_hook(
            make_feature_ablation_hook(h_feature_l, alpha)
        )
        model._ablation_hooks[l] = handle

def get_answers(verbal_idx, questions, vufa_alphas, out_file, iti_method, process_layers, prompt_type):
    print('will save to', out_file)
    batch_size = 32
    
    history_lens = defaultdict(int)
    # "alpha":  int
    if os.path.exists(out_file):
        with jsonlines.open(out_file, 'r') as reader:
            for line in reader:
                history_lens[float(line['alpha'])] += 1
    print("history_lens", history_lens)
    if prompt_type == 'uncertainty':
        sys_prompt = get_qa_system_prompt('uncertainty')
    for alpha in tqdm(vufa_alphas):
        print('alpha', alpha)
        register_feature_ablation_hook2(model, verbal_uncertain_feats, process_layers, alpha)

        history_len = history_lens[float(alpha)]
        all_message = [] 
        all_questions = []
        for qid, question in enumerate(questions):
            if qid < history_len:
                continue
            if prompt_type == 'uncertainty':
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Question: {question}\nAnswer: "},
                ]
            elif prompt_type == 'sentence':
                local_prompt = make_prompt('sentence', question)
                messages = [
                    {"role": "user", "content": local_prompt},
                ]
            all_message.append(messages)
            all_questions.append(question)
        generate_all_responses(model, tokenizer, all_questions, all_message, alpha, out_file, batch_size)


def generate_all_responses(model, tokenizer, all_questions, all_message, alpha, out_file, batch_size):
    for i in tqdm(range(0, len(all_message), batch_size), total=len(all_message)//batch_size):
        batch_message = all_message[i:i+batch_size]
        batch_question = all_questions[i:i+batch_size]

        inputs = tokenizer.apply_chat_template(
            batch_message, tokenize=True, add_generation_prompt=True, 
            truncation=True, padding=True,
            return_tensors="pt", return_dict=True).to(model.device)
        
        with torch.no_grad():
            # most_likely_answer
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
            )
            # responses
            N = 10
            try:
                outputs_responses = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=1,
                    num_return_sequences=N,
                )
            except:
                print("batch_message", batch_message)
                assert False
        decoded_answers = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        decoded_responses = tokenizer.batch_decode(outputs_responses[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        assert len(decoded_answers) == len(batch_question) == len(decoded_responses)//N
        for i, (question, answer) in enumerate(zip(batch_question, decoded_answers)):
            line = {'alpha': alpha,
                    'question': question, 
                    'most_likely_answer': answer,
                    'responses':decoded_responses[i*N:(i+1)*N],
                    }
            with jsonlines.open(out_file, 'a') as writer:
                writer.write(line)
    

def main(args):
    global model, tokenizer, device
    global questions_certain, questions_uncertain
    global verbal_uncertain_feats, verbal_uncertain_idx, verbal_certain_idx

    dataset = args.dataset
    split = args.split
    iti_method = args.iti_method
    process_layers = args.process_layers
    prompt_type = args.prompt_type
    model_name = args.model_name
    # vufa_alphas = np.arange(-2., 2.1, 0.5)
    vufa_alphas = [args.alpha]

    if iti_method == 5:
        output_dir = f"outputs2/{dataset}/{model_name}/{prompt_type}/{split}"
    else:
        output_dir = f"outputs/{dataset}/{model_name}/{prompt_type}/{split}"
    if args.run_certain:
        # run vuf ablation on **certain** examples with varying intervention strengths
        torch.cuda.empty_cache()
        if iti_method in [2, 5]:
            out_file = f"{output_dir}/questions_certain_with_vufi_{iti_method}_{args.str_process_layers}_{args.alpha}.jsonl"
        elif iti_method == 1:
            out_file = f"{output_dir}/questions_certain_with_vufi_{iti_method}_{args.dataset2}_{args.str_process_layers}_{args.alpha}.jsonl"
        print('out_file', out_file)
        get_answers(verbal_uncertain_idx, questions_certain, vufa_alphas, out_file, iti_method, process_layers, prompt_type)
    
    if args.run_uncertain:
        # run vuf ablation on **uncertain** examples with varying intervention strengths
        torch.cuda.empty_cache()
        if iti_method in [2, 5]:
            out_file = f"{output_dir}/questions_uncertain_with_vufi_{iti_method}_{args.str_process_layers}_{args.alpha}.jsonl"
        elif iti_method == 1:
            out_file = f"{output_dir}/questions_uncertain_with_vufi_{iti_method}_{args.dataset2}_{args.str_process_layers}_{args.alpha}.jsonl"
        get_answers(verbal_certain_idx, questions_uncertain, vufa_alphas, out_file, iti_method, process_layers, prompt_type)

    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='trivia_qa')
    parser.add_argument("--dataset2", type=str, default='trivia_qa')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--run_certain", type=int, default=0)
    parser.add_argument("--run_uncertain", type=int, default=0)
    parser.add_argument("--iti_method", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--str_process_layers", type=str, default='')
    parser.add_argument("--prompt_type", type=str)
    parser.add_argument("--model_name", type=str) # 
    args = parser.parse_args()


    dataset = args.dataset
    split = args.split
    prompt_type = args.prompt_type
    model_name =  args.model_name
    if 'Llama' in model_name:
        full_model_name = f'meta-llama/{model_name}'
    elif 'Qwen' in model_name:
        full_model_name = f'Qwen/{model_name}'
    elif 'Mistral' in model_name:
        full_model_name = f'mistralai/{model_name}'
    elif 'llama-3.1-8B-grpo' in model_name:
        full_model_name = f'AlistairPullen/{model_name}'
    args.process_layers = process_layers_to_process(args.str_process_layers)

    results_df = pd.read_csv(f"{root_path}/datasets/{dataset}/{model_name}/{split}.csv")
    questions = results_df['question'].tolist()
    vu_scores_llm = results_df['verbal_uncertainty'].to_numpy()
    answerable_labels = len(vu_scores_llm) * [1]

    # use threshold
    verbal_uncertain_idx = np.array([
        i for i in range(len(results_df))
        if answerable_labels[i] == 1 and vu_scores_llm[i] >= 0.9
    ])
    verbal_certain_idx = np.array([
        i for i in range(len(results_df))
        if answerable_labels[i] == 1 and vu_scores_llm[i] <= 0.05
    ])
    print(len(verbal_uncertain_idx), len(verbal_certain_idx))

    if args.iti_method == 2:
        verbal_uncertain_feats = torch.load(f'{current_dir}/outputs/merged/{model_name}/{prompt_type}/Hs_hedge_universal.pt')
    elif args.iti_method == 5: # use use
        verbal_uncertain_feats = torch.load(f'{current_dir}/outputs2/merged/{model_name}/{prompt_type}/Hs_hedge_universal.pt')
    elif args.iti_method == 1: # test generalization
        verbal_uncertain_feats = torch.load(f'{current_dir}/outputs/{args.dataset2}/{model_name}/{prompt_type}/train/Hs_hedge_universal.pt')

    questions_uncertain = [questions[i] for i in verbal_uncertain_idx]
    questions_certain = [questions[i] for i in verbal_certain_idx]

    ### load `LM and tokenizer ###
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(
        full_model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    main(args)