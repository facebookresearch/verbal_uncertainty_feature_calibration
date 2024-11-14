import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from os.path import join
import argparse
import itertools
import pickle
import json

import os
os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'


#################
hedge_resource_dir = "/data/home/ncan/data/hedging-resources"
hedge_word_file = "hedge_words.txt"
discourse_marker_file = "discourse_markers.txt"
booster_words_file = "booster_words.txt"

esu_filename = "expression_subjective_uncertainty.llama3.1.405B.txt"
euu_filename = "expression_universal_uncertainty.llama3.1.405B.txt"
#################

evaled_model_names = [
    'meta-llama/Meta-Llama-3.1-8B-Instruct', 
    # 'meta-llama/Meta-Llama-3.1-70B-Instruct'
]
judge_model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
judge_model_name_short = judge_model_name.split('/')[1]

dataset_names = [
    'nq_open', 'trivia_qa', 'pop_qa',
    'SelfAware', 'KUQ'
]

n_chunk = 10
chunk_idx = list(range(n_chunk))
slurm_arr_args = list(itertools.product(evaled_model_names, dataset_names, chunk_idx))


def main(args):
    ##### load expression embedding model (a fine-tuned Mistral-7B by Nvidia) #####
    emb_model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True).to('cuda')
    ##########################################

    ##### load and encode esu/euu hedging expressions #####
    with open(f"{hedge_resource_dir}/{esu_filename}") as f:
        esu_strings = f.readlines()
    with open(f"{hedge_resource_dir}/{euu_filename}") as f:
        euu_strings = f.readlines()
    
    esu_embeddings = F.normalize(emb_model.encode(esu_strings, instruction="", max_length=512), p=2, dim=1)
    euu_embeddings = F.normalize(emb_model.encode(euu_strings, instruction="", max_length=512), p=2, dim=1)
    ##########################################

    ##### load model generated answers #####
    evaled_model_name, ds_name, chunk_id = slurm_arr_args[args.job_id]
    evaled_model_name_short = evaled_model_name.split('/')[-1]
    qa_ds_fn = f"{evaled_model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}.json"
    with open(join(args.results_dir, qa_ds_fn), 'r') as f:
        qa_ds = json.load(f)
    ##########################################

    ##### take first 100 result examples #####
    # qa_ds = qa_ds[:10]
    ##########################################

    ##### get mean cosine sim scores to pre-defined hedging expressions #####
    lu_scores_esu, lu_scores_euu = [], []
    for example in tqdm(qa_ds):
        with torch.no_grad():
            model_answers = example['model answers']
            model_answer_embs = F.normalize(emb_model.encode(
                model_answers, instruction = "", max_length=512
            ), p=2, dim=1)

            esu_cos_sims = (model_answer_embs @ esu_embeddings.T).mean().cpu().item()
            euu_cos_sims = (model_answer_embs @ euu_embeddings.T).mean().cpu().item()

        lu_scores_esu.append(esu_cos_sims)
        lu_scores_euu.append(euu_cos_sims)
    ##########################################

    ### save judge results ###
    results_fn_esu = f"{evaled_model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}_lu-emb-sim-esu.p"
    results_fn_euu = f"{evaled_model_name_short}_{ds_name}_{chunk_id}_{args.prompt_method}_{args.temperature}_lu-emb-sim-euu.p"
    with open(join(args.results_dir, results_fn_esu), 'wb') as f:
        pickle.dump(lu_scores_esu, f)
    with open(join(args.results_dir, results_fn_euu), 'wb') as f:
        pickle.dump(lu_scores_euu, f)
    #############################


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='/data/home/jadeleiyu/mechanistic-uncertainty-calibrate/LUF/qa-evaluate/qa-eval-results', type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--n_response_per_question', default=20, type=int)
    parser.add_argument('--job_id', default=0, type=int)
    parser.add_argument('--prompt_method', default='uncertainty', type=str)
    
    args = parser.parse_args()
    main(args)

