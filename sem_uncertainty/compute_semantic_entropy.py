# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import jsonlines
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
sys.path.append(f'{root_path}/sem_uncertainty/')
from semantic_entropy.semantic_entropy import get_semantic_ids, cluster_assignment_entropy, EntailmentLlama, EntailmentVLLM
from semantic_entropy import utils
import torch
import argparse
# set seed
torch.manual_seed(42)
np.random.seed(42)
from tqdm import tqdm
utils.setup_logger()
sys.path.append(f'{root_path}')
from src.eval_utils import get_available_servers

def main(args):
    dataset = args.dataset
    split = args.split
    model_name  = args.model_name

    output_base_dir = f"{root_path}/sem_uncertainty/outputs/{dataset}/sentence/{model_name}"
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
    result_dict = dict()
    if os.path.exists(out_path):
        with open(out_path, "rb") as infile:
            result_dict = pickle.load(infile)
            if not (result_dict and result_dict['uncertainty_measures']):
                result_dict = dict()

    if result_dict:
        history_i = len(result_dict['uncertainty_measures']['cluster_assignment_entropy'])
        entropies = result_dict['uncertainty_measures'] # type_of_entropy -> list
        for e_type, e_list in entropies.items():
            assert len(e_list) == history_i
        assert len(result_dict['semantic_ids']) == history_i
    else:
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
    # parser.add_argument('--prompt_type', type=str, choices=["default", 'ignore_vu'])
    args = parser.parse_args()

    server_dict = get_available_servers()["meta-llama/Llama-3.1-70B-Instruct"]
    server_urls = server_dict["server_urls"]
    if "http" not in args.port:
        args.port = server_urls[int(args.port)]
    main(args)