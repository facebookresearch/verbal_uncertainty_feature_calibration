import requests
import re
import time
import json
import csv
import os
import bz2
import pickle
import _pickle as cPickle

try:
    import openai
    from openai import OpenAI
except:
    print("openai not installed")
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, LlamaTokenizer, LlamaForCausalLM
# from internal_information.modeling_llama import LlamaForCausalLM

FULL_TASKS_LIST = ["Code_to_Text", "Data_to_Text", "Dialogue_Generation", "Explanation", "Grammar_Error_Correction", "Number_Conversion", "Overlap_Extraction", "Paraphrasing", "Preposition_Prediction", "Program_Execution", "Question_Answering", "Sentence_Compression", "Summarization", "Text_to_Code", "Title_Generation", "Translation"]

def generate_step_hkust(prompt, temperature=0.2, model="gpt-35-turbo"):
    openai.api_base = "https://hkust.azure-api.net"
    openai.api_key = "5b12fcaaae4c45e6ba4174182daa5fe0" #zjiad
    # openai.api_key = "7e5fbeed936445378adde4ad4e723373" # eeziwei
    openai.api_type = "azure"
    openai.api_version = "2023-12-01-preview"
    # print(openai.api_key)
    fail_time = 0
    while fail_time < 5:
        if fail_time:
          print(fail_time)
        try:
            response = openai.ChatCompletion.create(
                engine=model,
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                temperature=temperature,
            )
            try:
              return response['choices'][0]['message']['content']
            except:
              print(response)
        except Exception as e:
            print(e)
            fail_time += 1
            time.sleep(5*fail_time)
    assert False
    

def generate_step(prompt, temperature=0.2, model="gpt-4-0125-preview"):
    url = "http://ecs.sv.us.alles-apin.openxlab.org.cn/v1/openai/v1/text/chat"
    headers = {"content-type": "application/json",
          "alles-apin-token":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjo1MSwidXNlcm5hbWUiOiJqaXppd2VpIiwiYXBwbHlfYXQiOjE2OTMzMDc2MDk4NzIsImV4cCI6MTc4ODE3NDgwODA5Nn0.4c72UArterJFLLYHsm4zNoHv73poeaArv993mRVqg7M"
    }

    fail_time = 0
    while fail_time < 50:
        print(fail_time)
        payload = {
          "model": model,
          "prompt": prompt,
          "role_meta": {
            "user_name": "user",
            "bot_name": "assistant"},
          "messages": [
            # {
            #   "sender_type": "user",
            #   "text": "Please write an algorithm in python"
            # ,
            # {
            #   "sender_type": "assistant",
            #   "text": "OK"
            # 
          ],
          "type": "json",
          "temperature": temperature,
          }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        try:
            reply = response.json()
        except:
            print(response)
            fail_time += 1
            time.sleep(fail_time)
            continue
        try:
            return reply["data"]["choices"][0]["message"]["content"]
        except:
            print(reply)
            if "The response was filtered due to the prompt triggering" in reply["data"]:
                return 'ethical issue'
            fail_time += 1
            time.sleep(fail_time)

    assert False

def generate_step_deepinfra(prompt, model="meta-llama/Meta-Llama-3-8B-Instruct"):
    openai = OpenAI(
        api_key="OWAib6bKB9QOOP5FnQe4k2E6rgHsSBMG",
        base_url="https://api.deepinfra.com/v1/openai",
    )

    fail_time = 0
    while fail_time < 5:
        chat_completion = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        try:
            return chat_completion.choices[0].message.content
        except:
            print(chat_completion)
            fail_time += 1
            time.sleep(0.1*fail_time)
    assert False

def process_qa(text):
    q1_res = re.search("Question 1: ", text).span()
    a1_res = re.search("Answer 1: ", text).span()
    q1 = text[q1_res[1]:a1_res[0]].strip()
    q2_res = re.search("Question 2: ", text).span()
    a1 = text[a1_res[1]:q2_res[0]].strip()
    a2_res = re.search("Answer 2: ", text).span()
    q2 = text[q2_res[1]:a2_res[0]].strip()
    q3_res = re.search("Question 3: ", text).span()
    a2 = text[a2_res[1]:q3_res[0]].strip()
    a3_res = re.search("Answer 3: ", text).span()
    q3 = text[q3_res[1]:a3_res[0]].strip()
    a3 = text[a3_res[1]:].strip()

    return [q1, q2, q3], [a1, a2, a3]


def get_lens(path, tokenizer):
    lens = []
    if path.endswith("csv"):
        with open(path) as f:
            reader = csv.reader(f)
            next(reader, None)
            for line in reader:
                question = line[1]
                lens.append(len(tokenizer.tokenize(question)))
    elif path.endswith("jsonl"):
        with jsonlines.open(path) as f:
            for l in f:
                qs, ans = process_qa(l["reply"])
                for q in enumerate(qs):
                    lens.append(len(tokenizer.tokenize(q[1])))
                    
    return lens

def get_llama_tokenizer():
    # model_path = "/cpfs01/shared/public/public_hdd/llmeval/model_weights/hf_hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
    model_path = "/share/jiziwei/Llama-2-7b-chat-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    return tokenizer

def init_model(model_name, device, padding_side="left", load_model=True):
    print("device", device)
    if model_name == 'Llama2-7B':
        model_path = "/share/jiziwei/Llama-2-7b-hf"
        tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side=padding_side)
        if load_model:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") #
    if model_name == 'Llama2-13B-Chat':
        model_path = "/share/jiziwei/Llama-2-13b-chat-hf"
        tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side=padding_side)
        if load_model:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") #
    elif model_name == "Llama2-7B-Chat":
        if os.path.isdir("/share/jiziwei/Llama-2-7b-chat-hf"): #116
            model_path = "/share/jiziwei/Llama-2-7b-chat-hf"
        elif os.path.isdir("/home/ziwei/Llama-2-7b-chat-hf"): #114
            model_path = "/home/ziwei/Llama-2-7b-chat-hf"
        tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side=padding_side)
        if load_model:
            model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto") #
    elif model_name == "Llama-3.1-8B-Instruct" or "llama3.1-8b_lora_sft" in model_name:
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

def if_source_summary(generated_path):
    sums = ['aeslc', 'multi_news', "samsum", "ag_news_subset", "newsroom", "gem_wiki_lingua_english_en", "cnn_dailymail", "opinion_abstracts_idebate", "opinion_abstracts_rotten_tomatoes", "huggingface", "gigaword"]
    sums += ["Summarization"]
    return any([s in generated_path.split("/") for s in sums])

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


def get_batch_generate(prompts, model, tokenizer, max_new_tokens, max_token=2048):
    if max_token == -1:
        max_token = float("inf")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    # print("input_length", input_length)
    if input_length < max_token:
        generated_texts = sub_batch_greedy_generate(inputs, input_length, model, tokenizer, max_new_tokens)
    else:
        generated_texts = []
        for i, prompt in enumerate(prompts):
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]
            if input_length < max_token:
                generated_texts += sub_batch_greedy_generate(inputs, input_length, model, tokenizer, max_new_tokens)
            else:
                print(f"INPUT:\n{prompt}\ntoo long")
                generated_texts.append("")

    return generated_texts


def sub_batch_greedy_generate(inputs, input_length, model, tokenizer, max_new_tokens):
    with torch.no_grad():
        generated_ids = model.generate(**inputs, 
                                        num_beams=1, do_sample=False, top_p=1.0, temperature=1.0,
                                        max_new_tokens=max_new_tokens)
        generated_ids = generated_ids[:, input_length:]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts




def compressed_pickle(file_path, data):
    with bz2.BZ2File(file_path, "w") as f:
        cPickle.dump(data, f)

def decompress_pickle(file_path):
    data = bz2.BZ2File(file_path, "rb")
    data = cPickle.load(data)
    return data
