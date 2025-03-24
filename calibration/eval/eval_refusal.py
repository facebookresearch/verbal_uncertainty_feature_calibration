import pickle
import argparse
import sys

import jsonlines
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_path)
from src import refusal
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--prompt_type', type=str, default="uncertainty")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--port', type=str, default="")
    parser.add_argument('--use_predicted', type=int, default=0)
    parser.add_argument('--allresponses', action='store_true')
    parser.add_argument('--max_alpha', type=float, default=1.0)
    parser.add_argument('--iti_method', type=int, default=2)
    parser.add_argument('--str_process_layers', type=str, default="")
    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    model_name = args.model_name
    prompt_type = args.prompt_type
    max_alpha = args.max_alpha
    iti_method = args.iti_method
    str_process_layers = args.str_process_layers
    print(f"Running refusal rate for {dataset} {split}")

    if args.use_predicted:
        output_base_dir = f"{root_path}/calibration/predicted_outputs/{dataset}/{model_name}/{prompt_type}/{split}"
    else:
        output_base_dir = f"{root_path}/calibration/outputs/{dataset}/{model_name}/{prompt_type}/{split}"
    
    if 'old' in prompt_type:
        input_path = f"{root_path}/calibration/outputs/{dataset}/Meta-Llama-3.1-8B-Instruct/uncertainty_old/{split}/with_vufi_2_range(15,32)_1.0.jsonl"
    else:
        if split == 'test': # semantic control
            if iti_method in [0,2]:
                input_path = f'{output_base_dir}/with_vufi_{iti_method}_{str_process_layers}_{max_alpha}.jsonl'
            elif iti_method == 1:
                input_path = f'{output_base_dir}/with_vufi_{iti_method}_trivia_qa_{str_process_layers}_{max_alpha}.jsonl'
            

    results_fn = input_path.replace("with_vufi", 'refusal')[:-1]
    print('input_path', input_path)
    print('results_fn', results_fn)

    with jsonlines.open(input_path) as f:
        all_lines = []
        for line in f:
            question = line['question']
            r = line['most_likely_answer']
            if r:
                all_lines.append({"question": question, "answer": r})
                
    refusal.run_eval(all_lines, results_fn, port=args.port, overwrite=False)
    with open(results_fn) as f:
        refusal_res = json.load(f)["refusal"]

    
    out_put = {}
    with jsonlines.open(input_path) as f:
        i = 0
        for line in f:
            question = line['question']
            r = line['most_likely_answer']
            if r:
                out_put[question] = refusal_res[i]
                i += 1
            else:
                out_put[question] = -1

        assert i == len(refusal_res)
    with open(results_fn, 'w') as f:
        json.dump(out_put, f)