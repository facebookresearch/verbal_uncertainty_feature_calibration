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
    parser.add_argument("--model_name", type=str) # 
    parser.add_argument("--N", type=int, default=-1) # 5000/-1
    args = parser.parse_args()

    # merge all dataset
    models_all_Hs_l_2d, models_n_uncertain = {}, {}
    DATASETS = {'trivia_qa': 'TriviaQA', 'nq_open': 'NQ-open', 'pop_qa': 'PopQA'}
    N = args.N
    prompt_type = args.prompt_type
    model_name = args.model_name
    # model_name = 'Meta-Llama-3.1-8B-Instruct'
    # model_name = 'Mistral-7B-Instruct-v0.3'
    # model_name = 'Qwen2.5-7B-Instruct'

    all_Hs_questions_uncertain_verbal, all_Hs_questions_certain_verbal = [], []
    all_vu_scores_llm_uncertain, all_vu_scores_llm_certain = [], []
    for dataset in ['trivia_qa', 'pop_qa', 'nq_open']:
        for split in ['val', 'test', 'train']:
            out_dir = f'{root_path}/calibration/outputs/{dataset}/{model_name}/{prompt_type}/{split}'
            Hs_questions_uncertain_verbal = torch.load(f'{out_dir}/uncertain_verbal.pt')
            Hs_questions_certain_verbal = torch.load(f'{out_dir}/certain_verbal.pt')
            print(Hs_questions_certain_verbal.shape)
            all_Hs_questions_uncertain_verbal.append(Hs_questions_uncertain_verbal)
            all_Hs_questions_certain_verbal.append(Hs_questions_certain_verbal)
            results_df = pd.read_csv(f"{root_path}/datasets/{dataset}/{model_name}/{split}.csv")
            vu_scores_llm = results_df['verbal_uncertainty'].to_numpy()
            per_all_vu_scores_llm_uncertain = [vu_scores_llm[i] for i in range(len(results_df)) if vu_scores_llm[i] >= 0.9]
            all_vu_scores_llm_uncertain += per_all_vu_scores_llm_uncertain
            per_all_vu_scores_llm_certain = [vu_scores_llm[i] for i in range(len(results_df)) if vu_scores_llm[i] <= 0.05]
            all_vu_scores_llm_certain += per_all_vu_scores_llm_certain
            print(len(per_all_vu_scores_llm_uncertain), len(per_all_vu_scores_llm_certain))
            print(Hs_questions_uncertain_verbal.shape, Hs_questions_certain_verbal.shape)
            assert len(per_all_vu_scores_llm_uncertain) == Hs_questions_uncertain_verbal.shape[0]
            assert len(per_all_vu_scores_llm_certain) == Hs_questions_certain_verbal.shape[0]

    all_Hs_questions_uncertain_verbal = torch.concat(all_Hs_questions_uncertain_verbal, dim=0)
    all_Hs_questions_certain_verbal = torch.concat(all_Hs_questions_certain_verbal, dim=0)
    print(all_Hs_questions_uncertain_verbal.shape)
    print(all_Hs_questions_certain_verbal.shape)

    if N > 0:
        verbal_uncertain_idx = np.argsort(all_vu_scores_llm_uncertain)[-N:]
        verbal_certain_idx = np.argsort(all_vu_scores_llm_certain)[:N]
        all_Hs_questions_uncertain_verbal = all_Hs_questions_uncertain_verbal[verbal_uncertain_idx]
        all_Hs_questions_certain_verbal = all_Hs_questions_certain_verbal[verbal_certain_idx]

    Hs_hedge_kuq = all_Hs_questions_uncertain_verbal.mean(0) - all_Hs_questions_certain_verbal.mean(0)

    os.makedirs(f'{root_path}/calibration/outputs/merged/{model_name}/{prompt_type}', exist_ok=True)
    torch.save(Hs_hedge_kuq, f'{root_path}/calibration/outputs/merged/{model_name}/{prompt_type}/Hs_hedge_universal.pt')

