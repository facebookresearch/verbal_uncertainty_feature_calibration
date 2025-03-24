import os
from os.path import join
home_path = os.path.expanduser("~")
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
set_seed(42)
from tqdm.auto import tqdm
import argparse
import sys
sys.path.append(current_dir)
from prompts import get_qa_system_prompt
import os
os.environ['HF_HOME'] = f'{home_path}/.cache/huggingface/'
import jsonlines
sys.path.append(root_path)
from sem_uncertainty.generate_answers import load_qa_ds

def prepare_inputs(tokenizer, batch_question, sys_prompt):
    batch_messages = []
    for question in batch_question:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question: {question}\nAnswer: "},
        ]
        batch_messages.append(messages)

    inputs = tokenizer.apply_chat_template(
        batch_messages, tokenize=True, add_generation_prompt=True, 
        return_tensors="pt", return_dict=True,
        padding=True,  # Enable padding
        truncation=True,  # Enable truncation
    )
    return inputs


def main_generate(args):
    dataset = args.dataset
    split = args.split
    model_name = args.model_name
    if args.model_name == 'Meta-Llama-3.1-8B-Instruct':
        model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    elif args.model_name == 'Mistral-7B-Instruct-v0.3':
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    elif args.model_name == 'Qwen2.5-7B-Instruct':
        model_name = 'Qwen/Qwen2.5-7B-Instruct'

    model_name_short = model_name.split('/')[-1]
    ### load LM and tokenizer ###
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    #############################

    ### load QA dataset ###
    qa_ds = load_qa_ds(dataset, split)
    
    ### get system prompt ###
    sys_prompt = get_qa_system_prompt(args.prompt_method)
    #############################

    results_fn = f"{model_name_short}_{dataset}_{split}_{args.prompt_method}_{args.temperature}.jsonl"
    out_root_dir = f"{current_dir}/{args.results_dir}"
    os.makedirs(out_root_dir, exist_ok=True)
    if os.path.exists(join(out_root_dir, results_fn)):
        with jsonlines.open(join(out_root_dir, results_fn), 'r') as f:
            history = list(f)
    else:
        with jsonlines.open(join(out_root_dir, results_fn), 'w') as f:
            history = []
    history_i = len(history)
    print(f"History exists. Start from {history_i}")

    ### generate answers ###
    all_questions = []
    all_examples = []
    for i, x in tqdm(enumerate(qa_ds), total=len(qa_ds)):
        if i < history_i:
            continue
        question = x['question']
        all_questions.append(question)
        all_examples.append(x)

    batch_size = 8
    n_response_per_question = args.n_response_per_question
    for i in tqdm(range(0, len(all_questions), batch_size)):
        batch_question = all_questions[i:i+batch_size]
        batch_example = all_examples[i:i+batch_size]

        inputs = prepare_inputs(tokenizer, batch_question, sys_prompt).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=args.temperature,
                num_return_sequences=n_response_per_question
            )
            answer_tokens = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True) # (n_res_per_q)
            answer_tokens = [ans.strip() for ans in answer_tokens]
            
        assert len(batch_example) * n_response_per_question == len(answer_tokens)
        for j, x in  enumerate(batch_example):
            x['model answers'] = answer_tokens[n_response_per_question*j:n_response_per_question*(j+1)]
            with jsonlines.open(join(out_root_dir, results_fn), 'a') as f:
                f.write(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='outputs', type=str)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--n_response_per_question', default=10, type=int)
    parser.add_argument('--prompt_method', default='uncertainty', type=str)
    parser.add_argument('--max_new_tokens', default=100, type=int)
    parser.add_argument('--dataset', default='trivia_qa', type=str)
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--model_name', default='Meta-Llama-3.1-8B-Instruct', type=str)
    args = parser.parse_args()
    main_generate(args)
    
    # add input_file and into args
    # args.input_file = ""
    # args.results_fn = ""
    # judge_main(args)
