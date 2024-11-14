# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
import sys
sys.path.append(f'{root_path}/sem_uncertainty/')
from semantic_entropy.huggingface_models import HuggingfaceModel
import openai
from tqdm.contrib.concurrent import thread_map
import subprocess

class VLLM:
    # HuggingfaceModel
    def __init__(self, name, max_new_tokens):
        self.name = 'meta-llama/Llama-3.1-70B-Instruct'
        self.port = name
        self.max_tokens = max_new_tokens

    def predict(self, prompt, temperature, output_hidden_states=False):
        client = openai.OpenAI(
            base_url=self.port,
            api_key="NOT A REAL KEY",
        )
        chat_completion = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user","content": prompt}],
            max_tokens=self.max_tokens,
            temperature=temperature,
        )

        return chat_completion.choices[0].message.content, '', ''
    
    def batch_predict(self, batch_prompts, temperature, output_hidden_states=False):
        all_return_values = thread_map(
            lambda p: self.predict(p, temperature=temperature),
            batch_prompts,
            max_workers=20,
            desc="using vllm")
        
        # all_return_values.append((sliced_answer, log_likelihoods, hidden_states))
        try:
            assert type(all_return_values) == list
            assert len(all_return_values[0]) == 3
            assert type(all_return_values[0][0]) == str
        except:
            print("batch_prompts", batch_prompts)
            print("all_return_values", type(all_return_values), all_return_values)
            assert False
        return all_return_values



def get_available_servers():
    agent_paths = {
        "llama3.1_70B": "meta-llama/Llama-3.1-70B-Instruct",
    }

    # Run squeue and capture output
    result = subprocess.run(['squeue', '--me', '-o', '"%j, %N, %T, %i"'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')

    # initialize server dict
    server_dict = {}
    
    # Iterate over each line, skipping the header
    for line in lines[1:]:
        line = line.strip('\"')
        # Get job name, nodelist, and status
        job_name, nodelist, status, job_id = line.split(', ')

        assert "[" not in nodelist, "Multi-node servers not currently supported."

        # keep only running jobs
        if status == "RUNNING" and job_name != "bash":
            try: 
                model_path = agent_paths[job_name]
                server_address = f"http://{nodelist}:8000/v1"

    
                if model_path in server_dict:
                    server_dict[model_path]["server_urls"].append(server_address)
                    server_dict[model_path]["job_ids"].append(job_id)
                else:
                    server_dict[model_path] = {"name": job_name, "server_urls": [server_address], "job_ids": [job_id]}
    
            except KeyError:
                continue
    # sort all server_urls and sort job_ids based on server_urls
    for model_path, server_info in server_dict.items():
        server_urls = server_info["server_urls"]
        job_ids = server_info["job_ids"]
        sorted_pairs = sorted(zip(server_urls, job_ids))
        sorted_server_urls, sorted_job_ids = zip(*sorted_pairs)
        sorted_server_urls = list(sorted_server_urls)
        sorted_job_ids = list(sorted_job_ids)
        server_info["server_urls"] = sorted_server_urls
        server_info["job_ids"] = sorted_job_ids
    return server_dict
