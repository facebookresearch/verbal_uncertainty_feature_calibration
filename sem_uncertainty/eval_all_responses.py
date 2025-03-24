"""Predict with LLM on task."""
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
import sys
import argparse
from tqdm import tqdm

from collections import defaultdict
sys.path.append(f'{root_path}/')
from sem_uncertainty.semantic_entropy.utils import batch_llm_metric
from src.eval_utils import VLLM
import pickle
import json



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--type', type=str, default="sentence")
    parser.add_argument('--port', type=str, default="")
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--prompt_type', type=str, choices=["default", 'ignore_vu'])
    args = parser.parse_args()

    # evaluator_model = HuggingfaceModel("Meta-Llama-3.1-70B-Instruct", max_new_tokens=50)
    evaluator_model = VLLM(args.port, max_new_tokens=50)
    dataset = args.dataset
    split = args.split
    batch_size = 40
    print(f"Running refusal rate for {dataset} {split}")

    output_base_dir = f"{root_path}/sem_uncertainty/outputs/{dataset}/{args.type}/{args.model_name}"
    res_path = f"{output_base_dir}/{split}_all_responses_acc.json"
    all_acc = defaultdict(list) # {qid: [acc1, acc2, ...]}
    if os.path.exists(res_path):
        with open(res_path, 'r') as f:
            all_acc = json.load(f)

    
    with open(f"{output_base_dir}/{split}_generations.pkl", 'rb') as f:
        all_predicted_answers, all_example, all_qid = [], [], []
        generations = pickle.load(f)
        for qid, example in generations.items():
            if qid in all_acc:
                if len(all_acc[qid]) == 10:
                    continue
                else:
                    # delete qid from all_acc
                    del all_acc[qid]
            for r in example['responses']:
                all_predicted_answers.append(r[0])
                all_example.append(example)
                all_qid.append(qid)

    for i in tqdm(range(0, len(all_predicted_answers), batch_size)):
        batch_predicted_answers = all_predicted_answers[i:i+batch_size]
        batched_example = all_example[i:i+batch_size]
        batched_qid = all_qid[i:i+batch_size]

        acces = batch_llm_metric(batch_predicted_answers, batched_example, evaluator_model, prompt_type=args.prompt_type)
        assert len(acces) == len(batch_predicted_answers) == len(batched_example) == len(batched_qid)
        for j, acc in enumerate(acces):
            all_acc[batched_qid[j]].append(acc)

        if i % batch_size*10 == 0:
            print(f"Processed {i} examples")
            with open(res_path, 'w') as f:
                json.dump(all_acc, f)

    with open(res_path, 'w') as f:
        json.dump(all_acc, f)