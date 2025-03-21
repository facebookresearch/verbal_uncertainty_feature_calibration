import pickle
import argparse
import sys
sys.path.append('/private/home/ziweiji/Hallu_Det/src')
sys.path.append('/home/ziweiji/Hallu_Det/src')
import refusal
import jsonlines
import os

"""

DATASETS=("trivia_qa" "nq_open" "pop_qa")
PORTS=('http://cr1-h100-p548xlarge-380:8000/v1' 'http://cr1-h100-p548xlarge-401:8000/v1' 'http://cr1-h100-p548xlarge-496:8000/v1')

length=${#DATASETS[@]}

# Loop through the arrays
for ((i=0; i<length; i++));
do
  DATA=${DATASETS[i]}
  PORT=${PORTS[i]}
  
  for SPLIT in "test" "val" "train"
    do
    python get_refusal_rate.py \
    --dataset $DATA \
    --split $SPLIT \
    --port $PORT \
    --type word
    done &

done

trivia_qa tmux 0  Qwen2.5-7B-Instruct 6008
trivia_qa tmux 2 Mistral-7B-Instruct-v0.3 6021
nq_open tmux 4 Qwen2.5-7B-Instruct 6024
nq_open tmux 6 Mistral-7B-Instruct-v0.3 6039
pop_qa tmux 8 Qwen2.5-7B-Instruct 6038
pop_qa tmux 10 Mistral-7B-Instruct-v0.3 6035



for D in 'nq_open'
do
for SPLIT in train test val
do
for MODEL in Mistral-7B-Instruct-v0.3
do
python ~/Hallu_Det/sem_uncertainty/get_refusal_rate.py \
--dataset $D \
--split $SPLIT \
--model_name $MODEL \
--port 'http://learnfair6039:8000/v1' &

done
done
done


"""




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

    if os.path.exists(f"/home/ziweiji/Hallu_Det/"):
        output_base_dir = f"/home/ziweiji/Hallu_Det/sem_uncertainty/outputs/{dataset}/{args.type}/{args.model_name}"
    else:
        output_base_dir = f"/private/home/ziweiji/Hallu_Det/sem_uncertainty/outputs/{dataset}/{args.type}/{args.model_name}"
    if args.allresponses:
        res_path = f"{output_base_dir}/{split}_all_responses_refusal_rate.json"
        T = 1.0
    else:
        res_path = f"{output_base_dir}/{split}_refusal_rate.json"
        T = 0.1
    print('res_path', res_path)

    # with open(f"{output_base_dir}/{split}_generations.pkl", 'rb') as f:
    #     all_lines = []
    #     generations = pickle.load(f)
    #     for qid, results in generations.items():
    #         question = results['question']
    #         if args.allresponses:
    #             for r in results['responses']:
    #                 r = r[0]
    #                 all_lines.append({"qid":qid, "question": question, "answer": r})
    #         else:
    #             r = results['most_likely_answer']['response']
    #             all_lines.append({"qid":qid, "question": question, "answer": r})
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