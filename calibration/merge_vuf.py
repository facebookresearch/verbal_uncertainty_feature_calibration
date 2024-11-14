# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import pandas as pd
import numpy as np
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_type", type=str, default='uncertainty') #'sentence' 'uncertainty'
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--score", type=str, default='verbal_uncertainty')
    parser.add_argument("--datasets", type=str, default='trivia_qa,pop_qa,nq_open')
    parser.add_argument("--N", type=int, default=-1) # 5000/-1
    args = parser.parse_args()

    # merge all dataset
    models_all_Hs_l_2d, models_n_uncertain = {}, {}
    N = args.N
    prompt_type = args.prompt_type
    model_name = args.model_name
    # model_name = 'Meta-Llama-3.1-8B-Instruct'
    # model_name = 'Mistral-7B-Instruct-v0.3'
    # model_name = 'Qwen2.5-7B-Instruct'

    all_Hs_questions_uncertain_verbal, all_Hs_questions_certain_verbal = [], []
    all_vu_scores_llm_uncertain, all_vu_scores_llm_certain = [], []
    for dataset in args.datasets.split(','):
        print(f"dataset: {dataset}")
        for split in ['val', 'test', 'train']:
            if args.score == 'verbal_uncertainty':
                out_dir = f'{root_path}/calibration/outputs/{dataset}/{model_name}/{prompt_type}/{split}'
            else:
                out_dir = f'{root_path}/calibration/outputs2/{dataset}/{model_name}/{prompt_type}/{split}'
            Hs_questions_uncertain_verbal = torch.load(f'{out_dir}/uncertain_verbal.pt')
            Hs_questions_certain_verbal = torch.load(f'{out_dir}/certain_verbal.pt')
            all_Hs_questions_uncertain_verbal.append(Hs_questions_uncertain_verbal)
            all_Hs_questions_certain_verbal.append(Hs_questions_certain_verbal)

            results_df = pd.read_csv(f"{root_path}/datasets/{dataset}/{model_name}/{split}.csv")
            vu_scores_llm = results_df[args.score].to_numpy()
            if args.score == 'verbal_uncertainty':
                per_all_vu_scores_llm_uncertain = [vu_scores_llm[i] for i in range(len(results_df)) if vu_scores_llm[i] >= 0.9]
                per_all_vu_scores_llm_certain = [vu_scores_llm[i] for i in range(len(results_df)) if vu_scores_llm[i] <= 0.05]
            else:
                per_all_vu_scores_llm_uncertain = list(vu_scores_llm)
                per_all_vu_scores_llm_certain = list(vu_scores_llm)

            all_vu_scores_llm_uncertain += per_all_vu_scores_llm_uncertain
            all_vu_scores_llm_certain += per_all_vu_scores_llm_certain

            print(len(per_all_vu_scores_llm_uncertain), Hs_questions_uncertain_verbal.shape)
            print(len(per_all_vu_scores_llm_certain), Hs_questions_certain_verbal.shape)
            assert len(per_all_vu_scores_llm_uncertain) == Hs_questions_uncertain_verbal.shape[0]
            assert len(per_all_vu_scores_llm_certain) == Hs_questions_certain_verbal.shape[0]

    all_Hs_questions_uncertain_verbal = torch.concat(all_Hs_questions_uncertain_verbal, dim=0)
    all_Hs_questions_certain_verbal = torch.concat(all_Hs_questions_certain_verbal, dim=0)
    print("all_Hs_questions_uncertain_verbal", all_Hs_questions_uncertain_verbal.shape)
    print("all_Hs_questions_certain_verbal", all_Hs_questions_certain_verbal.shape)

    if N > 0:
        verbal_uncertain_idx = np.argsort(all_vu_scores_llm_uncertain)[-N:]
        verbal_certain_idx = np.argsort(all_vu_scores_llm_certain)[:N]
        all_Hs_questions_uncertain_verbal = all_Hs_questions_uncertain_verbal[verbal_uncertain_idx]
        all_Hs_questions_certain_verbal = all_Hs_questions_certain_verbal[verbal_certain_idx]

    Hs_hedge_kuq = all_Hs_questions_uncertain_verbal.mean(0) - all_Hs_questions_certain_verbal.mean(0)
    if args.score == 'verbal_uncertainty':
        output_dir = f"{root_path}/calibration/outputs/merged/"
    else:
        output_dir = f"{root_path}/calibration/outputs2/merged/"
    os.makedirs(f'{output_dir}/{model_name}/{prompt_type}', exist_ok=True)
    torch.save(Hs_hedge_kuq, f'{output_dir}/{model_name}/{prompt_type}/Hs_hedge_universal.pt')
