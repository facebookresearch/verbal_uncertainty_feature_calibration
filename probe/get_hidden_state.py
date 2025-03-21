import os
import argparse
import re
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
import sys
sys.path.append(root_path)
from src.utils import init_model, process_layers_to_process
import random
random.seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import argparse
import time
import gc

def get_batch_internal_info(ids, texts, model, tokenizer, model_name,
                            layers_to_process, info_type='last',
                            save_hidden_state=False,
                            save_max_activation_ratio=False,
                            save_sparsity=False,
                            save_activation_correlation=False,
                            save_logits=False, 
                            save_attention=False, 
                            save_attention_lookback=False,
                            ignore_nan=False,
                            select_vocab=None,
                            max_length=500,
                            remove_question=False,
                            use_prompt=False,):
    internal_info = {}
    # if save_logits:
    #     early_exit_layers = layers_to_process
    # else:
    #     early_exit_layers = None
        
    # if re.search("((Llama)|(Mistral)).*-\d+B-((Instruct)|(Chat))", model_name):
    prompts = []
    for text in texts:
        if type(text) == dict:
            if 'only_question' in info_type:
                messages = [{"role": "user", "content": text["question"]},]
            else:
                messages = [{"role": "user", "content": text["question"]},
                            {"role": "assistant", "content": text["model_generated"]}]
        elif type(text) == str:
            messages = [{"role": "user", "content": text}]
        else:
            assert False
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prompts.append(prompt)
    if save_attention:
        inputs = tokenizer(prompts, padding='max_length', return_tensors="pt", max_length=max_length, truncation=True)
    else:
        inputs = tokenizer(prompts, padding=True, return_tensors="pt", max_length=max_length, truncation=True)
    inputs =  inputs.to(model.device)
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model(**inputs,
                        output_hidden_states=(save_hidden_state or save_max_activation_ratio or save_sparsity or save_activation_correlation), 
                        output_attentions=(save_attention or save_attention_lookback),)


    if save_hidden_state:
        hidden_state = defaultdict(dict) # {id: {layer: hidden_state}}
        if 'each' in info_type:
            mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.hidden_states[0].shape)
            mask = mask.float()# shape == (batch, seq_len, 4096)
            mask_sum = mask.sum(dim=1) if info_type == 'mean' else None
            
        for layer in layers_to_process:
            tmp = outputs.hidden_states[layer] # [batch, sequence_len, 4096]
            if 'last' in info_type: # token
                all_last_token_hidden_states_layer = tmp[:, -2, :]
            elif 'each' in info_type: # token
                # calculate mean of token's hidden states without padding
                mask_tmp = tmp * mask
                # if info_type == 'mean':
                #     all_mean_token_hidden_states_layer = mask_tmp.sum(dim=1) / mask_sum
                all_token_hidden_states_layer = mask_tmp
            else:
                raise ValueError(f"{info_type} is wrong info_type")
            # last/mean.shape == (len(ids), 4096), each.shape == (len(ids), seq_len, 4096)
            for bidx, id in enumerate(ids): # len(ids) == batch_size
                if 'last' in info_type:
                    h = all_last_token_hidden_states_layer[bidx] # (4096,)
                    # h = all_mean_token_hidden_states_layer[bidx] # (4096,) elif info_type == 'mean':
                elif 'each' in info_type:
                    h = all_token_hidden_states_layer[bidx] # (seq_len, 4096)
                    # remove the padding where the mask is 0
                    h = h[torch.sum(mask[bidx], dim=1) > 0]
                    
                if torch.isnan(h).all():
                    print("nan", layer, input_length)
                    print("id", id)
                    if ignore_nan:
                        hidden_state[id][layer] = h.cpu().half() # float 16
                        continue
                    else:
                        assert False
                else:
                    hidden_state[id][layer] = h.cpu().half() # float 16
        internal_info['hidden_state'] = hidden_state

    if save_max_activation_ratio or save_sparsity:
        # time1 = time.time()
        max_activation_ratio = defaultdict(dict) # {id: {layer: ratio}}
        sparsity = defaultdict(dict) # {id: {layer: sp}}
        for layer in layers_to_process:
            tmp = outputs.hidden_states[layer] # [batch, sequence_len, 4096]
            assert not torch.isnan(tmp).any(), f"hidden_states is nan, layer: {layer}"
            # calculate mean of token's hidden states without padding
            mask = inputs.attention_mask.unsqueeze(-1).expand(tmp.shape)
            mask = mask.float() # shape == (batch, seq_len, 4096)
            # assert not torch.isnan(mask).any(), f"mask is nan, layer: {layer}"
            mask_tmp = tmp * mask
            # assert not torch.isnan(mask_tmp).any(), f"mask_tmp is nan, layer: {layer}"
            
            max_activation = torch.max(mask_tmp, dim=2).values  # shape: (batch, seq_len)
            average_activation = torch.mean(mask_tmp, dim=2)  # shape: (batch, seq_len)
            # if average_activation = 0, average_activation = 1e-8
            average_activation = torch.where(average_activation == 0, torch.tensor(1e-8).to(average_activation), average_activation)
            max_activation_ratio_batch = torch.mean(max_activation / average_activation, dim=1)  # shape: (batch,)
            assert not torch.isnan(max_activation_ratio_batch).any(), f"max_activation_ratio_batch is nan, layer: {layer}"

            num_zero_activations = (mask_tmp == 0).sum(dim=2)  # shape: (batch, seq_len)
            hidden_dim = mask_tmp.shape[2]  # 4096
            sparsity_batch = torch.mean(num_zero_activations / hidden_dim, dim=1)  # shape: (batch,)

            for bidx, id in enumerate(ids):  # len(ids) == batch_size
                max_activation_ratio[id][layer] = max_activation_ratio_batch[bidx].cpu()  # 1
                sparsity[id][layer] = sparsity_batch[bidx].cpu()  # 1
                # assert not torch.isnan(max_activation_ratio[id][layer]), f"max_activation_ratio is nan, id: {id}, layer: {layer}"
                # assert not torch.isnan(sparsity[id][layer]), f"sparsity is nan, id: {id}, layer: {layer}"
                
        internal_info['max_activation_ratio'] = max_activation_ratio
        internal_info['sparsity'] = sparsity
        # time2 = time.time()
        # print("max_activation_ratio and sparsity time", time2-time1)

    if save_activation_correlation:
        # time1 = time.time()
        correlation = defaultdict(list) # {id: correlation}
        mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.hidden_states[0].shape).float()
        masked_hidden_states = [layer * mask for layer in outputs.hidden_states]
        valid_positions = torch.sum(mask, dim=2) > 0  # Shape: [batch_size, seq_len]

        # print("total layers", len(outputs.hidden_states)) # 33
        for layer_a_idx in range(32): # iterate over layers
            for layer_b_idx in range(layer_a_idx+1, 33):
                all_token_hidden_states_layer_a = masked_hidden_states[layer_a_idx]# [batch, sequence_len, 4096]
                all_token_hidden_states_layer_b = masked_hidden_states[layer_b_idx]

                for bidx, id in enumerate(ids): # len(ids) == batch_size
                    valid_idx = valid_positions[bidx]
                    h_a = all_token_hidden_states_layer_a[bidx][valid_idx] # (seq_len, 4096)
                    h_b = all_token_hidden_states_layer_b[bidx][valid_idx] # (seq_len, 4096)
                    corrs = compute_correlation(h_a, h_b)
                    correlation[id].append(corrs) # [4096]

        for id, corrs in correlation.items():
            correlation[id] = torch.mean(torch.stack(corrs), dim=0).cpu() # mean over layers
            assert not torch.isnan(correlation[id]).any(), f"correlation is nan, id: {id}"
        internal_info['correlation'] = correlation  # 4096
        # time2 = time.time()
        # print("correlation time", time2-time1)

    if save_logits:
        if select_vocab:
            with open(select_vocab) as f:
                select_vocab_ids = f.readlines()
                select_vocab_ids = sorted([int(v.strip()) for v in select_vocab_ids])

        logits_dict, outputs = outputs
        out_logits, out_each_logits = defaultdict(dict), defaultdict(dict)

        for layer in layers_to_process:
            tmp = logits_dict[layer] # layer, [batch, sequence_len, 32001] -> [batch, sequence_len, 32001]
            if 'last' in info_type: # token
                all_last_token_logits_layer = tmp[:, -2, :] # [sequence_len, 32001vab]

            # calculate mean of token's hidden states without padding
            # mask = inputs.attention_mask.unsqueeze(-1).expand(tmp.shape)
            # mask = mask.float()
            # mask_tmp = tmp * mask
            # # all_token_logits_layer = mask_tmp
            # if info_type == 'mean':
            #     all_mean_token_logits_layer = mask_tmp.sum(dim=1) / mask.sum(dim=1)
            #     all_mean_token_logits_layer = all_mean_token_logits_layer
            
            # last/mean.shape == (len(ids), 32001), each.shape == (len(ids), seq_len, 32001)
            for bidx, id in enumerate(ids): # len(ids) == batch_size
                # out_each_logits[id][layer] = all_token_logits_layer[bidx] # each

                if 'last' in info_type:
                    l = all_last_token_logits_layer[bidx]
                # elif info_type == 'mean':
                #     l = all_mean_token_logits_layer[bidx]
                # elif info_type == 'each':
                #     l = all_token_logits_layer[bidx]
                else:
                    raise ValueError(f"{info_type} is wrong info_type")
                if torch.isnan(l).all():
                    print("nan", layer, input_length)
                    print("id", id)
                    if ignore_nan:
                        if select_vocab:
                            out_logits[id][layer] = l[select_vocab_ids]
                        else:
                            out_logits[id][layer] = l
                        continue
                    else:
                        assert False
                else:
                    if select_vocab:
                        out_logits[id][layer] = l[select_vocab_ids]
                    else:
                        out_logits[id][layer] = l
                    
        internal_info['logits'] = out_logits
        internal_info['each_logits'] = out_each_logits
        del l, all_last_token_logits_layer, tmp

    if save_attention:
        # OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation
        attentions = defaultdict(dict) # {id: layer: last_layer_attention} 
        for layer in layers_to_process:
            tmp = outputs.attentions[layer-1] # [batch, num_heads, sequence_len, sequence_len]
            assert len(ids) == inputs.input_ids.shape[0]
            for bidx, (id, input_id) in enumerate(zip(ids, inputs.input_ids)):
                all_layer_attention = tmp[bidx] # [num_heads, sequence_len, sequence_len]
                doc_end_token_idx, generated_start_token_idx = find_index_of_sublist(input_id.tolist())
                all_layer_attention = all_layer_attention[:, generated_start_token_idx:, :] # [num_heads, num_generated_tokens, sequence_len]
                if remove_question:
                    doc_part = all_layer_attention[:, :, :doc_end_token_idx] 
                    ans_part = all_layer_attention[:, :, generated_start_token_idx:]
                    all_layer_attention = torch.cat([doc_part, ans_part], dim=-1) # [num_heads, num_generated_tokens, sequence_len]
                all_layer_attention = torch.max(all_layer_attention, dim=0).values # Max in multi heads # [num_generated_tokens, sequence_len]
                attentions[id][layer] = all_layer_attention # [num_generated_tokens, sequence_len]
        
        internal_info['attention'] = attentions
    if save_attention_lookback:
        # Lookback Lens
        lookback_ratio = defaultdict(dict) # {id: layer: ratio} 
        for layer in layers_to_process:
            if layer == 0:
                continue
            tmp = outputs.attentions[layer-1] # [batch, num_heads, sequence_len, sequence_len]
            
            for bidx, (id, input_id) in enumerate(zip(ids, inputs.input_ids)):
                if torch.isnan(input_id).any():
                    print("nan input_id", id)
                    assert False
                doc_end_token_idx, generated_start_token_idx = find_index_of_sublist(input_id.tolist())
                my_list = []
                for token_i in range(generated_start_token_idx, len(input_id)): # iterate on generated tokens
                    if remove_question:
                        attn_on_context = tmp[bidx, :, token_i, :doc_end_token_idx].mean(-1)
                    else:
                        attn_on_context = tmp[bidx, :, token_i, :generated_start_token_idx-4].mean(-1)
                    attn_on_new_tokens = tmp[bidx, :, token_i, generated_start_token_idx:token_i+1].mean(-1)
                    ratio = attn_on_context / (attn_on_context + attn_on_new_tokens) # [num_heads]
                    if torch.isnan(ratio).any():
                        print("nan ratio", id, layer)
                        assert False
                    my_list.append(ratio)
                lookback_ratio[id][layer] = torch.mean(torch.stack(my_list), dim=0) # [num_heads] mean over generated tokens
        internal_info['lookback_ratio'] = lookback_ratio       

    return internal_info


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


def prepare_internal_info(source_dirs, layers_to_process, internal_model_name, batch_size, device, 
                        info_type, 
                        save_hidden_state, save_max_activation_ratio, save_sparsity, save_activation_correlation,
                        save_logits, save_attention, save_attention_lookback,
                        splits=["train", "val", "test"],
                        select_vocab=None,
                        max_length=500,
                        remove_question=False,
                        save_cache="",
                        use_prompt=False):
    
    assert type(source_dirs) == list
    print("prepare internal_info source_dirs", source_dirs)
    time1 = time.time()
    
    all_hidden_states, all_max_activation_ratio, all_sparsity, all_correlation = {}, {}, {}, {}
    all_logits, all_each_logits, all_attentions, all_lookback_ratios = {}, {}, {}, {}
    
    source_paths = set()
    for source_dir in source_dirs:
        for f in os.listdir(source_dir):
            if f.endswith("csv") and any([s in f for s in splits]):
                source_paths.add(f"{source_dir}/{f}")
                
    for source_path in source_paths:
        print("source_path", source_path)
        assert os.path.isfile(source_path) and source_path.endswith("csv")

        each_source_hidden_states, each_source_max_activation_ratio, each_source_sparsity, each_source_correlation = {}, {}, {}, {}
        each_source_logits, each_source_each_logits, each_source_attentions, each_source_lookback_ratios = {}, {}, {}, {}
        if save_cache:
            # source_path = /home/zjiad/Hallu_Det/datasets/ANAH/data_s/test_comprehensive.csv
            save_dir = "/".join(source_path.split("/")[:-1])
            split = source_path.split("/")[-1].split("_")[0]
            split = split.split(".csv")[0]
            save_path = f"{save_dir}/{save_cache}/{split}/"
            if os.path.isdir(save_path):
                history_ids = [f[:-3] for f in os.listdir(save_path)]
                print("load from cache", save_path)
            else:
                history_ids = []
                os.makedirs(save_path, exist_ok=True)
                print("cache not found", save_path)
            internal_model, internal_tokenizer = init_model(internal_model_name, device, "left")
            internal_tokenizer.padding_side = "left"
        if not each_source_hidden_states:
            print("start to prepare_internal_info for", source_path)
            dataset = pd.read_csv(source_path, encoding="utf8")
            for i in tqdm(range(0, len(dataset), batch_size)):
                ids, texts = [], []
                for j in range(i, min(i+batch_size, len(dataset))):
                    idx = str(dataset.loc[j, "id"])
                    if idx not in history_ids:
                        ids.append(idx)
                        if "text" in dataset.columns:
                            texts.append(dataset.loc[j, 'text'])
                        elif "question" in dataset.columns and "model_generated" in dataset.columns:
                            texts.append({"question": dataset.loc[j, 'question'], "model_generated": dataset.loc[j, 'model_generated']})
                        elif "question" in dataset.columns:
                            texts.append({"question": dataset.loc[j, 'question']})
                        else:
                            assert False, "text or question and answer should be in the columns"
                if ids:
                    print("prepare_internal_info for", len(ids), "samples")
                    batch_internal_info = get_batch_internal_info(ids, texts, internal_model, internal_tokenizer, internal_model_name,
                                    layers_to_process, info_type,
                                    save_hidden_state, save_max_activation_ratio, save_sparsity, save_activation_correlation,
                                    save_logits, save_attention, save_attention_lookback,
                                    ignore_nan=False,
                                    select_vocab=select_vocab,
                                    max_length=max_length,
                                    remove_question=remove_question,
                                    use_prompt=use_prompt)
                    if save_cache and save_hidden_state:
                        save_path = f"{save_dir}/{save_cache}/{split}/"
                        os.makedirs(save_path, exist_ok=True)
                        batch_hidden_state = batch_internal_info['hidden_state']
                        for idx, activation in batch_hidden_state.items():
                            torch.save(activation, save_path + f"{idx}.pt")
                    if save_hidden_state:
                        batch_hidden_state = batch_internal_info['hidden_state'] # {id: {layer: hidden_state}}
                        each_source_hidden_states.update(batch_hidden_state)
                    if save_max_activation_ratio:
                        batch_max_activation_ratio = batch_internal_info['max_activation_ratio']
                        each_source_max_activation_ratio.update(batch_max_activation_ratio)
                    if save_sparsity:
                        batch_sparsity = batch_internal_info['sparsity']
                        each_source_sparsity.update(batch_sparsity)
                    if save_activation_correlation:
                        batch_correlation = batch_internal_info['correlation']
                        each_source_correlation.update(batch_correlation)
                    if save_logits:
                        batch_logits = batch_internal_info['logits'] # {id: {layer: logits}}
                        each_source_logits.update(batch_logits)
                        batch_each_logits = batch_internal_info['each_logits'] # {id: {layer: logits}}
                        each_source_each_logits.update(batch_each_logits)
                    if save_attention:
                        batch_attention = batch_internal_info['attention']
                        each_source_attentions.update(batch_attention)
                    if save_attention_lookback:
                        batch_lookback_ratio = batch_internal_info['lookback_ratio']
                        each_source_lookback_ratios.update(batch_lookback_ratio)
            
        all_hidden_states.update(each_source_hidden_states)
        all_max_activation_ratio.update(each_source_max_activation_ratio)
        all_sparsity.update(each_source_sparsity)
        all_correlation.update(each_source_correlation)
        all_logits.update(each_source_logits)
        all_each_logits.update(each_source_each_logits)
        all_attentions.update(each_source_attentions)
        all_lookback_ratios.update(each_source_lookback_ratios)


    if save_hidden_state:    
        print("shape of hidden_states", list(list(all_hidden_states.values())[0].values())[0].shape)
    if save_max_activation_ratio:
        print("shape of max_activation_ratio", list(list(all_max_activation_ratio.values())[0].values())[0], list(list(all_max_activation_ratio.values())[0].values())[0].shape) # [1]
    if save_sparsity:
        print("shape of sparsity", list(list(all_sparsity.values())[0].values())[0].shape) # [1]
    if save_activation_correlation:
        print("shape of correlation", list(all_correlation.values())[0].shape) # [4096]
    if save_logits:
        print("shape of logits", list(list(all_logits.values())[0].values())[0].shape)   # [sequence_len, 32001]
    if save_attention:
        print("shape of attentions", list(list(all_attentions.values())[0].values())[0].shape) # [generated_sequence_len, generated_sequence_len]
    if save_attention_lookback:
        print("shape of lookback_ratios", list(list(all_lookback_ratios.values())[0].values())[0].shape)
    time2 = time.time()
    print("prepare internal_info time", time2-time1)
    # js_divs = KV_DIV_logits(layers_to_process, all_each_logits)
    js_divs = {}
    if internal_model:
        del internal_model, internal_tokenizer
    gc.collect()
    torch.cuda.empty_cache() 

    return {"hidden_states": all_hidden_states, 
            "max_activation_ratio": all_max_activation_ratio, "sparsity": all_sparsity, "activation_correlation": all_correlation,
            "logits": all_logits, "js_divs": js_divs, 
            'attentions': all_attentions, 'lookback_ratios': all_lookback_ratios}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=11008)
    parser.add_argument("--model_type", type=str, default="LlamaMLP")
    parser.add_argument("--p_num_latents_list", nargs='*', type=int) 
    parser.add_argument("--splits", nargs='*', type=str)
    parser.add_argument("--share_perceiver", action="store_true")
    parser.add_argument("--info_type", type=str, choices=["last", "mean", 'each', 'only_question_each', 'only_question_last'], default="each")
    parser.add_argument("--cache_info", action="store_true")
    parser.add_argument("--layers_to_process", nargs='*', type=str)
    parser.add_argument("--hidden_state_dims", type=str)
    parser.add_argument("--select_hidden_state_dims_method", type=str)
    parser.add_argument("--uncertainty", type=str, default="")
    parser.add_argument("--annealing_step", type=int, default=10) #?????
    parser.add_argument("--lrs", nargs='*', type=float)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--training_epoch", type=int, default=10)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--report_to",  type=str, default="wandb")
    parser.add_argument("--load_best_model_at_end", type=bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="accuracy")
    parser.add_argument("--greater_is_better", type=bool, default=True)
    parser.add_argument("--metric_name", type=str, default="accuracy")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--hidden_batch_size", type=int, default=64)
    parser.add_argument("--source_dirs", nargs='*', type=str) 
    parser.add_argument("--data_paths", nargs='*', type=str)
    parser.add_argument("--label", type=str, default='label')
    parser.add_argument("--internal_model_name", type=str, default="")
    parser.add_argument("--save_cache", type=str, default="")
    parser.add_argument("--save_dir_root", type=str)
    parser.add_argument("--clean_checkpoints", action="store_true")
    parser.add_argument("--only_predict", action="store_true")
    parser.add_argument("--ignore_missing_info", action="store_true")
    parser.add_argument("--save_hidden_state", action="store_true")
    parser.add_argument("--save_max_activation_ratio", action="store_true")
    parser.add_argument("--save_sparsity", action="store_true")
    parser.add_argument("--save_activation_correlation", action="store_true")
    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_attention", action="store_true")
    parser.add_argument("--save_attention_lookback", action="store_true")
    parser.add_argument("--remove_question", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--select_vocab", type=str)
    parser.add_argument("--input_x", default="")
    parser.add_argument("--predict_result_file", type=str, default="test_results.json")
    parser.add_argument("--max_seq_length", type=int, default=512)
    args = parser.parse_args()

    batch_size = args.batch_size
    # args.per_device_train_batch_size = args.per_device_batch_size
    # args.per_device_eval_batch_size = args.per_device_batch_size
    # args.gradient_accumulation_steps = batch_size//args.per_device_batch_size

    model_type = args.model_type
    if_resnet = "ResNet" in model_type
    info_type = args.info_type
    o_layers_to_process = args.layers_to_process
    args.layers_to_process = process_layers_to_process(args.layers_to_process)
    args.save_dir_root = re.sub("\/+", "/", args.save_dir_root)
    if args.save_dir_root[-1] == "/":
        args.save_dir_root = args.save_dir_root[:-1]
    save_dir_root = args.save_dir_root
    if args.input_x:
        INPUT = args.input_x
    elif args.save_hidden_state:
        INPUT = "hidden_states"
    elif args.save_max_activation_ratio and args.save_sparsity and args.save_activation_correlation:
        INPUT = "max_activation_ratio_sparsity_correlation"
    elif args.save_max_activation_ratio:
        # assert "LSTM" in args.classifer_type
        INPUT = "max_activation_ratio"
    elif args.save_sparsity:
        # assert "LSTM" in args.classifer_type
        INPUT = "sparsity"
    elif args.save_activation_correlation:
        INPUT = "activation_correlation"
    elif args.save_logits:
        INPUT = "logits"
    elif args.save_attention:
        INPUT = "attentions"
    elif args.save_attention_lookback:
        INPUT = "lookback_ratios"
    else:
        assert False
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    internal_model_name = args.internal_model_name
    save_attention = False
    print('device:', args.device)
    if not args.only_predict:
        all_internal_info = prepare_internal_info(args.source_dirs, args.layers_to_process, args.internal_model_name, args.hidden_batch_size, args.device, 
                            info_type, args.save_hidden_state, args.save_max_activation_ratio, args.save_sparsity, args.save_activation_correlation, args.save_logits, args.save_attention, args.save_attention_lookback,
                            max_length=args.max_seq_length, select_vocab=args.select_vocab, remove_question=args.remove_question, save_cache=args.save_cache,
                            splits=args.splits)
                            