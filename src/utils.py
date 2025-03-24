import os
import torch
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM

def init_model(model_name, device, padding_side="left", load_model=True):
    print("device", device)
    if "Llama-3.1-8B-Instruct" in model_name or "llama3.1-8b_lora_sft" in model_name:
        model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
        if load_model:
            if "llama3.1-8b_lora_sft" in model_name:
                model_path = model_name
            # device_string = f"{device.type}:{device.index}" if device.type == "cuda" else "cpu"
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") 

    elif model_name == "Llama-3.1-70B-Instruct":
        model_path = "meta-llama/Meta-Llama-3.1-70B-Instruct" 
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
        if load_model:
            # device_string = f"{device.type}:{device.index}" if device.type == "cuda" else "cpu"
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") 

    elif model_name == "Mistral-7B-Instruct-v0.3":
        model_path = "mistralai/Mistral-7B-Instruct-v0.3"
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
        if load_model:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    elif model_name == "Qwen2.5-7B-Instruct":
        model_path = 'Qwen/Qwen2.5-7B-Instruct'
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side)
        if load_model:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") 
    
    tokenizer.pad_token = tokenizer.eos_token 
    if load_model:
        model.resize_token_embeddings(len(tokenizer))
        # model.to(device)
    else:
        model = None
    return model, tokenizer


def process_layers_to_process(layers_to_process):
    if not layers_to_process:
        layers_to_process2 = []
    elif type(layers_to_process) == str and "range" in layers_to_process:
        layers_to_process2 = sorted(list(eval(layers_to_process)))
    elif len(layers_to_process) == 1 and "range" in layers_to_process[0]: #range33
        layers_to_process2 = sorted(list(eval(layers_to_process[0])))
    else:
        layers_to_process2 = sorted([int(x) for x in layers_to_process])
    return layers_to_process2
