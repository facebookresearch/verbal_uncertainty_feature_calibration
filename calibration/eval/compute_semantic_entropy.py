"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import jsonlines
import argparse
import sys
import torch
# set seed
torch.manual_seed(42)
np.random.seed(42)
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(f'{root_path}/sem_uncertainty/')
from semantic_entropy.semantic_entropy import get_semantic_ids, cluster_assignment_entropy, EntailmentLlama, EntailmentVLLM
from semantic_entropy.utils import setup_logger

setup_logger()

def main(args):
    dataset = args.dataset
    split = args.split
    max_alpha = args.max_alpha
    iti_method = args.iti_method
    model_name = args.model_name
    prompt_type = args.prompt_type
    str_process_layers = args.str_process_layers

    if args.use_predicted:
        output_base_dir = f"{root_path}/calibration/predicted_outputs/{dataset}/{model_name}/{prompt_type}/{split}"
    else:
        output_base_dir = f"{root_path}/calibration/outputs/{dataset}/{model_name}/{prompt_type}/{split}"
    
    if iti_method == 1:
        input_path = f'{output_base_dir}/with_vufi_{iti_method}_trivia_qa_{str_process_layers}_{max_alpha}.jsonl'
    elif iti_method in [0, 2]:
        input_path = f'{output_base_dir}/with_vufi_{iti_method}_{str_process_layers}_{max_alpha}.jsonl'

    results_fn = input_path.replace("with_vufi", 'uncertainty_measures')
    results_fn = results_fn.replace("jsonl", 'pkl')
    print(f"Results will be saved to {results_fn}")
    ############## load model #################
    if 'llama' in args.entailment_model.lower():
        entailment_model = EntailmentLlama(None, False, args.entailment_model, prompt_type='ignore_lu')
    elif 'http' in args.entailment_model.lower():
        entailment_model = EntailmentVLLM(None, False, args.entailment_model, prompt_type='ignore_lu')
    else:
        raise ValueError
    logging.info('Entailment model loading complete.')
        
    ############## history #################
    history_i = 0
    if os.path.exists(results_fn):
        with open(results_fn, "rb") as infile:
            result_dict = pickle.load(infile)
            history_i = len(result_dict['uncertainty_measures']['cluster_assignment_entropy'])
            entropies = result_dict['uncertainty_measures'] # type_of_entropy -> list
            for e_type, e_list in entropies.items():
                assert len(e_list) == history_i
        assert len(result_dict['semantic_ids']) == history_i
    else:
        result_dict = dict()
        result_dict['uncertainty_measures'] = dict()
        result_dict['semantic_ids'] = []
        entropies = defaultdict(list)
    print(f"History: {history_i}")    

    with jsonlines.open(input_path) as f:
        generations = list(f)
    print('len(generations)', len(generations))
    
    for idx, example in tqdm(enumerate(generations), total=len(generations)):
        if idx < history_i:
            continue
        responses = example["responses"]
        if not responses:
            # print("No responses")
            result_dict['semantic_ids'].append([])
            entropies['cluster_assignment_entropy'].append(-1)
            continue

        # Compute semantic ids.
        semantic_ids = get_semantic_ids(
            responses, model=entailment_model,
            strict_entailment=True, example=example)
        
        result_dict['semantic_ids'].append(semantic_ids)
        # Compute entropy from frequencies of cluster assignments.
        entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))
        torch.cuda.empty_cache()
        
        with open(results_fn, 'wb') as f:
            # type_of_entropy -> list
            result_dict['uncertainty_measures'].update(entropies)
            pickle.dump(result_dict, f)

    with open(results_fn, 'wb') as f:
        result_dict['uncertainty_measures'].update(entropies)
        pickle.dump(result_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--entailment_model', type=str, default="")
    parser.add_argument('--prompt_type', type=str,)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--max_alpha', type=float, default=1.0)
    parser.add_argument("--use_predicted", type=int, default=0)
    parser.add_argument("--iti_method", type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--str_process_layers', type=str, default='')
    args = parser.parse_args()
    main(args)