import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import argparse
import json
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
sys.path.append(root_path)
from verbal_uncertainty.prompts import get_qa_system_prompt
sys.path.append(f"{root_path}/sem_uncertainty/")
from semantic_entropy.utils import make_prompt
# set seed 
torch.manual_seed(42)
np.random.seed(42)
import jsonlines
from causal import register_feature_ablation_hook2, generate_all_responses
from src.utils import process_layers_to_process
from src.detection_utils import LLAMA_PROBE_PATHS

def get_answers(questions, alphas, detection_res, out_file, process_layers, model, prompt_type):
    if prompt_type == 'uncertainty':
        sys_prompt = get_qa_system_prompt('uncertainty')
    print('will save to', out_file)
    
    if os.path.exists(out_file):
        with jsonlines.open(out_file, 'r') as reader:
            history_len = len(list(reader))
    else:
        history_len = 0
    print("history_len", history_len)
    assert len(questions) == len(alphas) == len(detection_res)
    
    for i, (question, alpha, dr) in tqdm(enumerate(zip(questions, alphas, detection_res)), total=len(questions)):
        if i < history_len:
            continue

        if alpha == 0 or dr == 0: # don't change
            line = {'alpha': 0,
                'question': question, 
                'most_likely_answer': '',
                'responses': [],
                }
            with jsonlines.open(out_file, 'a') as writer:
                writer.write(line)
        else: # regenerate
            register_feature_ablation_hook2(model, verbal_uncertain_feats, process_layers, alpha)
            if prompt_type == 'uncertainty':
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Question: {question}\nAnswer: "}
                ]
            elif args.prompt_type == 'sentence':
                local_prompt = make_prompt('sentence', question)
                messages = [
                    {"role": "user", "content": local_prompt},
                ]
            generate_all_responses(model, tokenizer, [question], [messages], alpha, out_file, batch_size=1)
     
        if i % 100 == 0:
            torch.cuda.empty_cache()



def load_detection_res(dataset, model_name):
    path = f"{root_path}/detector/LR_outputs/{dataset}/{model_name}/verbal_uncertainty_sentence_semantic_entropy.json"
    with open(path) as f:
        detection_res = json.load(f)["y_pred"]
    return detection_res
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='trivia_qa')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--iti_method", type=int, default=2)
    parser.add_argument("--dataset2", type=str, default='trivia_qa')
    parser.add_argument("--str_process_layers", type=str, default='range(15,32)')
    parser.add_argument("--max_alpha", type=float, default=1.0)
    parser.add_argument("--use_predicted", type=int, default=0)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--prompt_type", type=str)
    args = parser.parse_args()


    dataset = args.dataset
    split = args.split
    model_name = args.model_name
    if 'Llama' in model_name:
        full_model_name = f'meta-llama/{model_name}'
        # args.str_process_layers = 'range(12,32)'
    elif 'Qwen' in model_name:
        full_model_name = f'Qwen/{model_name}'
        # args.str_process_layers = 'range(16,28)'
    elif 'Mistral' in model_name:
        full_model_name = f'mistralai/{model_name}'
        # args.str_process_layers = 'range(15,32)'
    process_layers = args.process_layers = process_layers_to_process(args.str_process_layers)
    prompt_type = args.prompt_type
    
    results_df = pd.read_csv(f"{root_path}/datasets/{dataset}/{model_name}/{split}.csv")
    
    if args.use_predicted:
        output_dir = f'{root_path}/calibration/predicted_outputs/{dataset}/{model_name}/{args.prompt_type}/{split}'
        with open(f"{root_path}/probe/"+LLAMA_PROBE_PATHS['verbal_uncertainty'][dataset]+f"/{dataset}_predict_results.json") as f:
            data = json.load(f)
            vu_scores_llm = np.array(data["predictions"])
        with open(f"{root_path}/probe/"+LLAMA_PROBE_PATHS['sentence_semantic_entropy'][dataset]+f"/{dataset}_predict_results.json") as f:
            data = json.load(f)
            su_scores = np.array(data["predictions"])
    else:
        output_dir = f'{root_path}/calibration/outputs/{dataset}/{model_name}/{args.prompt_type}/{split}'
        vu_scores_llm = results_df['verbal_uncertainty'].to_numpy()
        su_scores = results_df['sentence_semantic_entropy'].to_numpy()
    os.makedirs(output_dir, exist_ok=True)
    assert len(vu_scores_llm) == len(su_scores)
    print('vu_scores_llm', vu_scores_llm)
    # min 0 max su_scores 1
    # min 0 max su_scores MAX_SE
    # mean 0.8832655919376744 1.2997004660200655 0.6444195462553963
    # make sure su_scores from 0 to 0.5
    print('su_scores', su_scores)
    questions = results_df['question'].tolist()
    
    MAX_SE = 2.302585092994045
    MAX_ALPHA = args.max_alpha

    detection_res = load_detection_res(dataset, model_name)

    if args.iti_method == 0:
        alphas = (su_scores/MAX_SE) * MAX_ALPHA
        alphas = np.clip(alphas, 0, MAX_ALPHA)
        alphas = np.round(alphas, 4)
        verbal_uncertain_feats = torch.load(f'{root_path}/calibration/outputs/merged/{model_name}/{args.prompt_type}/Hs_hedge_universal.pt')
        out_file = f"{output_dir}/with_vufi_{args.iti_method}_{args.str_process_layers}_{args.max_alpha}.jsonl"
    elif args.iti_method == 1:
        # use other datasets
        out_file = f"{output_dir}/with_vufi_{args.iti_method}_{args.dataset2}_{args.str_process_layers}_{args.max_alpha}.jsonl"
        alphas = (su_scores/MAX_SE - vu_scores_llm) * MAX_ALPHA
        # max alphas and 0
        alphas = np.clip(alphas, 0, MAX_ALPHA)
        alphas = np.round(alphas, 4)
        print('alphas', alphas)
        verbal_uncertain_feats = torch.load(f'{root_path}/calibration/outputs/{args.dataset2}/{model_name}/{args.prompt_type}/train/Hs_hedge_universal.pt')
    elif args.iti_method == 2:
        out_file = f"{output_dir}/with_vufi_{args.iti_method}_{args.str_process_layers}_{args.max_alpha}.jsonl"
        alphas = (su_scores/MAX_SE - vu_scores_llm) * MAX_ALPHA
        # max alphas and 0
        alphas = np.clip(alphas, 0, MAX_ALPHA)
        alphas = np.round(alphas, 4)
        print('alphas', alphas)
        verbal_uncertain_feats = torch.load(f'{root_path}/calibration/outputs/merged/{model_name}/{args.prompt_type}/Hs_hedge_universal.pt')
    else:
        assert False

    # with open(f"{output_dir}/activation_vuf_mean_std_cache.pkl", "rb") as f:
    #     vuf_cache_all = pickle.load(f)
    # assert len(vuf_cache_all[0][0]) == len(results_df)
    #############################
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
    get_answers(questions, alphas, detection_res, out_file, process_layers, model, prompt_type)