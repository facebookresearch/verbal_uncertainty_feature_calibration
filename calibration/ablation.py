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
import pickle
from src.utils import process_layers_to_process
# set seed 
torch.manual_seed(42)
np.random.seed(42)
import jsonlines
from collections import defaultdict

# def remove_all_hooks(model):
#     """Remove all forward/backward hooks from the model."""
#     for module in model.modules():
#         module._forward_hooks.clear()
#         module._forward_pre_hooks.clear()
#         module._backward_hooks.clear()


# def register_feature_ablation_hook(model, Hs_feature, feat_vals_def, process_layers):
#     for l in process_layers:
#         device_idx_l = model.hf_device_map[f"model.layers.{l}"]
#         h_feature_l = Hs_feature[l].to(f"cuda:{device_idx_l}")  # (h_dim)
#         h_feature_l = h_feature_l / torch.sqrt(h_feature_l.pow(2).sum(-1))

#         def make_feature_ablation_hook(h_feature_l, feat_val_def):
#             def feature_ablation_hook(module, inputs, outputs):
#                 if isinstance(outputs, tuple): # this case
#                     outputs_0 = outputs[0]   # (B, seq_len, h_dim)
#                     if outputs_0.shape[1] > 1:
#                         b = torch.matmul(outputs_0, h_feature_l).unsqueeze(-1)
#                         outputs_0 -= h_feature_l * b
#                         outputs_0 += h_feature_l * feat_val_def
#                         # print('feat_val_def', feat_val_def)
#                         # print('mean', b.mean().item(), 'std', b.std().item(), 'b', b)
#                         # print('diff', feat_val_def-b.mean().item(), 'b',feat_val_def-b)
#                     return (outputs_0,) + outputs[1:]
#                 else:
#                     assert False
#                     if outputs.shape[1] > 1: # (B, seq_len, h_dim)
#                         outputs -= h_feature_l * torch.matmul(outputs, h_feature_l).unsqueeze(-1)
#                         outputs += feat_val_def * h_feature_l
#                     return outputs
#             return feature_ablation_hook

#         model.model.layers[l].register_forward_hook(
#             make_feature_ablation_hook(h_feature_l, feat_vals_def[l])
#         )
        
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

def get_answers(ling_idx, questions, lufa_alphas, out_file, iti_method, process_layers, prompt_type):
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
    for alpha in tqdm(lufa_alphas):
        print('alpha', alpha)
        register_feature_ablation_hook2(model, ling_uncertain_feats, process_layers, alpha)

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
    global ling_uncertain_feats, ling_uncertain_idx, ling_certain_idx

    dataset = args.dataset
    split = args.split
    iti_method = args.iti_method
    process_layers = args.process_layers
    prompt_type = args.prompt_type
    model_name = args.model_name
    # lufa_alphas = np.arange(-2., 2.1, 0.5)
    lufa_alphas = [args.alpha]

    output_dir = f"outputs/{dataset}/{model_name}/{prompt_type}/{split}"
    if args.run_certain:
        # run luf ablation on **certain** examples with varying intervention strengths
        torch.cuda.empty_cache()
        if iti_method == 2:
            out_file = f"{output_dir}/questions_certain_with_lufi_{iti_method}_{args.str_process_layers}_{args.alpha}.jsonl"
        elif iti_method == 1:
            out_file = f"{output_dir}/questions_certain_with_lufi_{iti_method}_{args.dataset2}_{args.str_process_layers}_{args.alpha}.jsonl"
        print('out_file', out_file)
        get_answers(ling_uncertain_idx, questions_certain, lufa_alphas, out_file, iti_method, process_layers, prompt_type)
    
    if args.run_uncertain:
        # run luf ablation on **uncertain** examples with varying intervention strengths
        torch.cuda.empty_cache()
        if iti_method == 2:
            out_file = f"{output_dir}/questions_uncertain_with_lufi_{iti_method}_{args.str_process_layers}_{args.alpha}.jsonl"
        elif iti_method == 1:
            out_file = f"{output_dir}/questions_uncertain_with_lufi_{iti_method}_{args.dataset2}_{args.str_process_layers}_{args.alpha}.jsonl"
        get_answers(ling_certain_idx, questions_uncertain, lufa_alphas, out_file, iti_method, process_layers, prompt_type)

    
        
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

    if model_name == 'Meta-Llama-3.1-8B-Instruct':
        results_df = pd.read_csv(f"{root_path}/datasets/{dataset}/sampled/{split}.csv")
    else:
        results_df = pd.read_csv(f"{root_path}/datasets/{dataset}/{model_name}/{split}.csv")
    questions = results_df['question'].tolist()
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

    if args.iti_method == 2:
        ling_uncertain_feats = torch.load(f'./outputs/merged/{model_name}/{prompt_type}/Hs_hedge_universal.pt')
    elif args.iti_method == 1: # test generalization
        ling_uncertain_feats = torch.load(f'./outputs/{args.dataset2}/{model_name}/{prompt_type}/train/Hs_hedge_universal.pt')

    questions_uncertain = [questions[i] for i in ling_uncertain_idx]
    questions_certain = [questions[i] for i in ling_certain_idx]

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