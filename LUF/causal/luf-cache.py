import json
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm.auto import tqdm
from os.path import join
import numpy as np
from ast import literal_eval
import argparse
import itertools
import pickle
from datasets import load_dataset
from sklearn.decomposition import PCA

import os
import json
os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'

import sys
sys.path.append('/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/LUF/')
from utils import get_qa_system_prompt

model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
dataset_names = [
    'nq_open', 'trivia_qa', 'pop_qa'
]


def get_ling_uncertainty_features(ds_name, model, tokenizer, device, results_merged, n_example_per_class=500):
    ds_idx = [i for i in range(len(results_merged)) if results_merged[i]['dataset name'] == ds_name]
    questions = [results_merged[i]['question'] for i in ds_idx]

    ling_uncertainty_scores = np.array([
        results_merged[i]['linguistic uncertainty'] for i in ds_idx
    ])
    ling_uncertain_idx = np.argsort(ling_uncertainty_scores)[-n_example_per_class:]
    ling_certain_idx = np.argsort(ling_uncertainty_scores)[:n_example_per_class]

    sys_prompt = get_qa_system_prompt('uncertainty')
    Hs_questions_uncertain_ling = []
    Hs_questions_certain_ling = []

    for i in tqdm(ling_uncertain_idx):
        question = questions[i]
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question: {question}\nAnswer: "},
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
        question = questions[i]
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question: {question}\nAnswer: "},
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
    ling_uncertain_feats = Hs_questions_uncertain_ling.mean(0) - Hs_questions_certain_ling.mean(0)

    return ling_uncertain_feats


def remove_all_hooks(model):
    """Remove all forward/backward hooks from the model."""
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()

def register_feature_ablation_hook(model, Hs_feature, feat_vals_def, layer_idx, alpha=1.0):
    for l in layer_idx:
        device_idx_l = device_idx_l = model.hf_device_map[f"model.layers.{l}"]
        h_feature_l = Hs_feature[l].to(f"cuda:{device_idx_l}")  # (h_dim)
        h_feature_l = h_feature_l / torch.sqrt(h_feature_l.pow(2).sum(-1))

        def make_feature_ablation_hook(h_feature_l, feat_val_def):
            def feature_ablation_hook(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    outputs_0 = outputs[0]   # (B, seq_len, h_dim)
                    if outputs_0.shape[1] > 1:
                        outputs_0 -= h_feature_l * torch.matmul(outputs_0, h_feature_l).unsqueeze(-1)
                        outputs_0 += feat_val_def * h_feature_l * alpha
                    return (outputs_0,) + outputs[1:]
                else:
                    if outputs.shape[1] > 1:
                        outputs -= h_feature_l * torch.matmul(outputs_0, h_feature_l).unsqueeze(-1)
                        outputs += feat_val_def * h_feature_l * alpha
                    return outputs
            return feature_ablation_hook

        model.model.layers[l].register_forward_hook(
            make_feature_ablation_hook(h_feature_l, feat_vals_def[l])
        )


def register_feature_projection_hook(model, Hs_feature, layer_idx, fp_cache):
    for l in layer_idx:
        device_idx_l = device_idx_l = model.hf_device_map[f"model.layers.{l}"]
        h_feature_l = Hs_feature[l].to(f"cuda:{device_idx_l}")  # (h_dim)
        h_feature_l = h_feature_l / torch.sqrt(h_feature_l.pow(2).sum(-1))

        def make_fp_hook(h_feature_l, l):
            def fp_hook(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    outputs_0 = outputs[0]   # (B, seq_len, h_dim)
                    if outputs_0.shape[1] > 1:
                        fp = torch.matmul(outputs_0[:, -1], h_feature_l).mean().cpu()
                        fp_cache[l].append(fp)
                    
                else:
                    if outputs.shape[1] > 1:
                        fp = torch.matmul(outputs[:, -1], h_feature_l).mean().cpu()
                        fp_cache[l].append(fp)
                    
            return fp_hook

        model.model.layers[l].register_forward_hook(make_fp_hook(h_feature_l, l))


def get_mean_activation_features(model, tokenizer, layers_to_fp, features, questions, sys_prompt):
    remove_all_hooks(model)
    torch.cuda.empty_cache()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    layers_to_fp = np.arange(28) if 'gemma' in model.config.model_type else np.arange(32)

    fp_cache = {l:[] for l in layers_to_fp}
    register_feature_projection_hook(
        model, features, layers_to_fp, fp_cache
    )

    for question in tqdm(questions):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question: {question}\nAnswer: "},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, 
            return_tensors="pt", return_dict=True
        )
        with torch.no_grad():
            model(**inputs.to('cuda'))
            torch.cuda.empty_cache()

    remove_all_hooks(model)
    torch.cuda.empty_cache()

    fp_cache = {
        l : (torch.stack(fps).mean().item(), torch.stack(fps).std().item())
        for l, fps in fp_cache.items()
    }

    return fp_cache


def main(args):
    job_id = args.job_id
    ### load LM and tokenizer ###
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval();
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #############################

    model_name_short = model_name.split('/')[-1]
    with open(f'qa-eval-results/{model_name_short}-ling-sem-uncertainty.json', 'r') as f:
        results_merged = json.load(f)

    ds_name = dataset_names[job_id]
    ds_idx = [i for i in range(len(results_merged)) if results_merged[i]['dataset name'] == ds_name]
    questions = [results_merged[i]['question'] for i in ds_idx]
    n_example_per_class = int(0.1 * len(questions))
    ling_uncertain_feats = get_ling_uncertainty_features(
        ds_name, model, tokenizer, device, results_merged,
        n_example_per_class=n_example_per_class
    )

    sys_prompt = get_qa_system_prompt(args.prompt_method)
    layers_to_fa = np.arange(28) if 'gemma' in model.config.model_type else np.arange(32)
    luf_cache = get_mean_activation_features(
        model, tokenizer, layers_to_fa, ling_uncertain_feats, questions, sys_prompt
    )
    torch.save(ling_uncertain_feats, f'activation-cache/ling_uncertain_feats_{model_name_short}_{ds_name}.pt')
    with open(f'activation-cache/luf_mean_std_cache_{model_name_short}_{ds_name}.p', 'wb') as f:
        pickle.dump(luf_cache, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', default='/data/home/jadeleiyu/hall-irrelevant-context/data', type=str)
    parser.add_argument('--job_id', default=0, type=int)
    parser.add_argument('--prompt_method', default='uncertainty', type=str)
    
    args = parser.parse_args()
    main(args)