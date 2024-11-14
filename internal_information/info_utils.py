import os
import re
import torch


hall_prompt = """Please generate a hallucinated answer that is plausible-sounding but factually incorrect to the following question:\n\n"""
justify_hall = "Please evaluate the above answer for any hallucinations, which are plausible-sounding but factually incorrect. Is there any hallucination in the answer?"

def get_internal_info_prompts(texts, use_prompt, tokenizer):
    # if re.search("((Llama)|(Mistral)).*-\d+B-((Instruct)|(Chat))", model_name):
    prompts = []
    for text in texts:
        if type(text) == dict:
            if use_prompt=='Classification':
                messages = [{"role": "user", "content": text["question"]},
                            {"role": "assistant", "content": text["answer"]},
                            {"role": "user", "content": justify_hall},]
            elif use_prompt=='HLM':
                messages = [{"role": "user", "content": hall_prompt + text["question"]},
                            {"role": "assistant", "content": text["answer"]},]
            elif not use_prompt:
                messages = [{"role": "user", "content": text["question"]},
                            {"role": "assistant", "content": text["answer"]}]
            else:
                assert False
        elif type(text) == str:
            messages = [{"role": "user", "content": text}]
        else:
            assert False
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prompts.append(prompt)
    return prompts



        # save_cache = self.args.save_cache
        # if save_cache:
        #     # source_path = /home/zjiad/Hallu_Det/datasets/ANAH/data_s/test_comprehensive.csv
        #     save_dir = "/".join(source_path.split("/")[:-1])
        #     split = source_path.split("/")[-1].split("_")[0]
        #     if "IDK" in save_dir and split != "train": # 
        #         # print("too large to cache")
        #         pass
        #     else:
        #         save_path = f"{save_dir}/{split}_{save_cache}.pt"
        #         if os.path.isfile(save_path):
        #             cached_hidden_state = torch.load(save_path) # {id: {layer: hidden_state}}


def compute_correlation(x, y):
    mean_x = torch.mean(x.float(), dim=0)
    mean_y = torch.mean(y.float(), dim=0)
    xm = x - mean_x
    ym = y - mean_y
    # Compute the numerator of the correlation coefficient in a more efficient manner
    r_num = torch.sum(xm * ym, dim=0)
    # Compute the denominator of the correlation coefficient more efficiently
    xm_norm = torch.sqrt(torch.sum(xm ** 2, dim=0))
    ym_norm = torch.sqrt(torch.sum(ym ** 2, dim=0))
    r_den = xm_norm * ym_norm
    # Compute correlation coefficients directly
    corrs = r_num / r_den
    return corrs

def find_index_of_sublist(list_A):
    res = []  # "Question:"      "Answer:"
    for i, sublist in enumerate([[14924, 25], [2170, 77, 6703, 25]]):
        start_index = 0
        start_indices = []
        for i in range(len(list_A)):
            if list_A[i:i+len(sublist)] == sublist:
                start_indices.append(start_index)
                start_index = i
        start_indices.append(start_index)
        if not start_indices[-1]:
            print("not found", sublist)
            with open("not_found.txt", "a") as f:
                f.write(str(list_A)+"\n")
        assert start_indices[-1]
        if i: # "Answer:"
            end_index = start_indices[-1]+len(sublist) # last one
            res.append(end_index)
        else: # "Question:"
            res.append(start_indices[-1]) # last one
    return res
