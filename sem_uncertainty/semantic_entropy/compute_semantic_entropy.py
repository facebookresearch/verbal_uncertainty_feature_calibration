"""Compute uncertainty measures after generating answers."""
# /home/ziweiji/Hallu_Det/sem_uncertainty/semantic_entropy/compute_uncertainty_measures.py
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import jsonlines

import sys

current_path = os.getcwd()
root_path = os.path.abspath(os.path.join(current_path, os.pardir))
from sem_uncertainty.utils.semantic_entropy import EntailmentLlama, EntailmentVLLM, cluster_assignment_entropy, get_semantic_ids
from sem_uncertainty.utils import utils

import torch
import argparse
# set seed
torch.manual_seed(42)
np.random.seed(42)



from tqdm import tqdm
utils.setup_logger()

def main(args):
    dataset = args.dataset
    split = args.split
    model_name  = args.model_name

    output_base_dir = f"{root_dir}/sem_uncertainty/outputs/{dataset}/sentence/{model_name}"
    input_path = f"{output_base_dir}/{split}_1.0.jsonl"
    print('input_path', input_path)
    assert not os.path.isdir(input_path)
    out_path = f"{output_base_dir}/{split}_semantic_entropy.pkl"

    print(f"Results will be saved to {out_path}")
    ############## load model #################
    if 'llama' in args.port.lower():
        port = EntailmentLlama(None, False, args.port, prompt_type='default')
    elif 'http' in args.port.lower():
        port = EntailmentVLLM(None, False, args.port, prompt_type='default')
    else:
        raise ValueError
    logging.info('Entailment model loading complete.')
        
    ############## history #################
    history_i = 0
    if os.path.exists(out_path):
        with open(out_path, "rb") as infile:
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
        responses = example["model answers"]
        assert len(responses) == 10

        # Compute semantic ids.
        semantic_ids = get_semantic_ids(
            responses, model=port,
            strict_entailment=True, example=example)
        
        result_dict['semantic_ids'].append(semantic_ids)
        # Compute entropy from frequencies of cluster assignments.
        entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))
        torch.cuda.empty_cache()
        
        with open(out_path, 'wb') as f:
            # type_of_entropy -> list
            result_dict['uncertainty_measures'].update(entropies)
            pickle.dump(result_dict, f)

    with open(out_path, 'wb') as f:
        result_dict['uncertainty_measures'].update(entropies)
        pickle.dump(result_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--port', type=str, default="")
    # parser.add_argument('--prompt_type', type=str, choices=["default", 'ignore_lu'])
    args = parser.parse_args()

    main(args)
"""

python /home/ziweiji/Hallu_Det/calibration/eval/compute_semantic_entropy.py  \
--dataset trivia_qa \
--split test \
--port  "Meta-Llama-3.1-70B-Instruct" \
--max_alpha 1.0 

"""
