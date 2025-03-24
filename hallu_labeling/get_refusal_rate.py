import os
import argparse
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
sys.path.append(root_path)
from src import refusal
import jsonlines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--type', type=str, default="sentence")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--port', type=str, default="")
    parser.add_argument('--allresponses', action='store_true')
    args = parser.parse_args()

    dataset = args.dataset
    split = args.split
    print(f"Running refusal rate for {dataset} {split}")

    output_base_dir = f"{root_path}/sem_uncertainty/outputs/{dataset}/{args.type}/{args.model_name}"
    if args.allresponses:
        res_path = f"{output_base_dir}/{split}_all_responses_refusal_rate.json"
        T = 1.0
    else:
        res_path = f"{output_base_dir}/{split}_refusal_rate.json"
        T = 0.1
    print('res_path', res_path)

    with jsonlines.open(f"{output_base_dir}/{split}_{T}.jsonl") as f:
        print(f"input is {output_base_dir}/{split}_{T}.jsonl")
        all_lines = []
        for line in f:
            qid = line['id']
            question = line['question']
            r = line['model answers']
            if args.allresponses:
                for r in r:
                    all_lines.append({"qid":qid, "question": question, "answer": r})
            else:
                assert len(r) == 1
                r = r[0]
                all_lines.append({"qid":qid, "question": question, "answer": r})
    print('all_lines', len(all_lines))

    refusal.run_eval(all_lines, res_path, port=args.port, overwrite=False)