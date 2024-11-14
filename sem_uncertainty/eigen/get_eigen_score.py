
# https://github.com/alibaba/eigenscore/blob/main/func/metric.py#L174

import sys
import os
cwd = os.getcwd()
root_path = "/home/ziweiji/Hallu_Det/"
sys.path.append(root_path)
import torch
import numpy as np
from tqdm.auto import tqdm
from internal_information.get_internal_info import get_batch_internal_info
from src.utils import init_model
import pickle
import argparse


def get_hidden_states(index, question, answer_list, internal_model, internal_tokenizer, internal_model_name, model_max_new_tokens):
    selected_layer = 16
    info_type = 'last'
    ids = [f"{index}_{i}" for i in range(len(answer_list))]
    texts = [{'question': question, 'model_generated': answer} for answer in answer_list]
    batch_internal_info = get_batch_internal_info(ids, texts, internal_model, internal_tokenizer, internal_model_name,
                                    [selected_layer], info_type,
                                    True, False, False, False,
                                    False, False, False,
                                    ignore_nan=False,
                                    select_vocab=None,
                                    max_length=model_max_new_tokens,
                                    remove_question=None,
                                    use_prompt=False)
    
    batch_hidden_state = batch_internal_info['hidden_state']
    last_embeddings = []
    for idx, activation in batch_hidden_state.items():
        last_token = activation[selected_layer]
        last_embeddings.append(last_token)
    last_embeddings = torch.stack(last_embeddings, dim=0)
    
    return last_embeddings # [num_seq, embedding_size]

###### 计算最后一个token特征的语义散度的作为句子的语义散度
###### 需要考虑每个句子的长度不一致，去除padding的token的影响
###### hidden_states : (num_tokens, num_layers, num_seq, num_input_tokens/1, embedding_size)
def getEigenIndicator_v0(last_embeddings): 
    alpha = 1e-3
    CovMatrix = torch.cov(last_embeddings).cpu().numpy().astype(float)
    u, s, vT = np.linalg.svd(CovMatrix+alpha*np.eye(CovMatrix.shape[0]))
    eigenIndicator = np.mean(np.log10(s))
    return eigenIndicator, s




def main(args):
    dataset = args.dataset
    type = args.type
    internal_model_name = args.internal_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    internal_model, internal_tokenizer = init_model(internal_model_name, device, "left")
    internal_tokenizer.padding_side = "left"

    for split in args.dataset_splits:
        print(f"Processing {dataset} {split}")
        with open(f'{root_path}/sem_uncertainty/outputs/{dataset}/{type}/{split}_generations.pkl', 'rb') as f:
            generations = pickle.load(f)
        first = 0
        all_results = {}
        for index, generation in tqdm(generations.items()):
            question = generation['question']
            answer_list = []
            for a in generation['responses']:
                answer_list.append(a[0])

            last_embeddings = get_hidden_states(index, question, answer_list, internal_model, internal_tokenizer, internal_model_name,
                                                args.model_max_new_tokens)
            
            
            eigenIndicator, s = getEigenIndicator_v0(last_embeddings)
            all_results[index] = {"eigenIndicator": eigenIndicator, "s": s}

            if first == 0:
                print('answer_list', len(answer_list))
                print("answer_list[0]", answer_list[0])
                print("last_embeddings.shape", last_embeddings.shape)
                print("eigenIndicator", eigenIndicator)
                print("s", s)
                first += 1

        with open(f'{root_path}/sem_uncertainty/outputs/{dataset}/{type}/{split}_eigen.pkl', 'wb') as f:
            pickle.dump(all_results, f)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_splits", nargs="*")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--type", type=str, default="")
    parser.add_argument("--model_max_new_tokens", type=int, default=500)
    parser.add_argument("--internal_model_name", type=str, default="Llama-3.1-8B-Instruct")
    args = parser.parse_args()
    main(args)
