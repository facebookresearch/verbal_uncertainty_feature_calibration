import os
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
import json
import jsonlines
import argparse
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)

def merge_most_likely_answer(dataset, model, split):
    type = 'sentence' #'word'

    out_path = f"{root_path}/datasets/{dataset}/{model}_{type}/{split}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    input_path = f"{root_path}/datasets/{dataset}/sampled/{split}.csv"
    df = pd.read_csv(input_path)
    out_dir = f"{root_path}/sem_uncertainty/outputs/{dataset}/{type}/{model}/"
    path = f"{out_dir}/{split}_0.1.jsonl"

    if os.path.exists(path):
        refusal_path = f"{out_dir}/{split}_refusal_rate.json"
        with open(refusal_path) as f:
            refusal = json.load(f)['refusal']
        print("refusal rate", np.mean(refusal))
        acc_path = f"{out_dir}/{split}_most_likely_acc.json"
        with open(acc_path) as f:
            accuracy = json.load(f) # dict

        
        with jsonlines.open(path) as reader:
            data = list(reader)
            if len(data) == len(df) == len(refusal):
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    id = str(row['id'])
                    response =  data[i]['model answers']
                    assert len(response) == 1
                    response = response[0]

                    if refusal[i] or accuracy[id]:
                        l = 'ok'
                    else:
                        l = 'hallucinated'
                    df.at[i, 'model_generated'] = response
                    df.at[i, f'label'] = l
                    df.at[i, 'accuracy'] = accuracy[id]
                    df.at[i, 'refusal'] = refusal[i]
                columns_to_keep = ['id', 'question', 'model_generated', 'label', 'accuracy', 'refusal']
                df = df[columns_to_keep]
                df.to_csv(out_path, index=False)
            else:
                print(f"Length mismatch: {dataset}, {split} {len(data)} vs {len(df)} vs {len(refusal)}")
    else:
        print(f"File not found: {path}")



def merge_verbal_uncertainty(dataset, model, split):
    out_path = input_path = f"{root_path}/datasets/{dataset}/{model}/{split}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(input_path):
        input_path = f"{root_path}/datasets/{dataset}/sampled/{split}.csv"
    df = pd.read_csv(input_path)
    path = f"{root_path}/verbal_uncertainty/outputs/{model}_{dataset}_{split}_uncertainty_1.0_vu-llm-judge.json"
    if os.path.exists(path):
        with open(path, "rb") as reader:
            data = json.load(reader)
            # assert len(data) == len(df)
            if len(data) == len(df):
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    assert len(data[i]) == 10
                    verbal_uncertainty = np.mean([x for x in data[i] if x !=-1])
                    df.at[i, 'verbal_uncertainty'] = verbal_uncertainty

                df.to_csv(out_path, index=False)
            else:
                print(f"Length mismatch: {dataset}, {split} {len(data)} vs {len(df)}")
    else:
        print(f"File not found: {path}")



def merge_semantic_uncertainty(dataset, model, split):
    out_path = input_path = f"{root_path}/datasets/{dataset}/{model}/{split}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(input_path):
        input_path = f"{root_path}/datasets/{dataset}/sampled/{split}.csv"
    df = pd.read_csv(input_path)

    type = 'sentence' #'no_refuse_word', 'no_refuse_sentence',
    path = f"{root_path}/sem_uncertainty/outputs/{dataset}/{type}/{model}/{split}_semantic_entropy.pkl"
    if os.path.exists(path):
        with open(path, "rb") as reader:
            data = pickle.load(reader)
            data = data['uncertainty_measures']['cluster_assignment_entropy']
            # assert len(data) == len(df)
            if len(data) == len(df):
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    df.at[i, f'{type}_semantic_entropy'] = data[i]
                df.to_csv(out_path, index=False)
            else:
                print(f"Length mismatch: {dataset}{model}, {split} {len(data)} vs {len(df)}")
    else:
        print(f"File not found: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='trivia_qa')
    parser.add_argument("--model", type=str, default='Meta-Llama-3.1-8B-Instruct')
    args = parser.parse_args()
    #  dataset in ['trivia_qa', 'pop_qa', 'nq_open']:
    #  model in ['Meta-Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B-Instruct']:
      
    dataset = args.dataset
    model = args.model
    for split in ['test', 'val', 'train']:
        merge_most_likely_answer(dataset, model, split)
        merge_verbal_uncertainty(dataset, model, split)
        merge_semantic_uncertainty(dataset, model, split)