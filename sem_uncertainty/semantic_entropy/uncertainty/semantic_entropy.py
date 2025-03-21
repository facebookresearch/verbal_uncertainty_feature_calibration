"""
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.
Copyright (c) 2024 OATML

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import pickle
import logging

import random
import numpy as np
import wandb
import openai
import torch
import torch.nn.functional as F

from uncertainty.huggingface_models import HuggingfaceModel
from uncertainty.utils import openai as oai
from uncertainty.utils import utils
import collections

random.seed(10)
from tqdm.contrib.concurrent import thread_map

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    def save_prediction_cache(self):
        pass

class EntailmentLLM(BaseEntailment):

    entailment_file = 'entailment_cache.pkl'

    def __init__(self, entailment_cache_id, entailment_cache_only):
        self.prediction_cache = self.init_prediction_cache(entailment_cache_id)
        self.entailment_cache_only = entailment_cache_only

    def init_prediction_cache(self, entailment_cache_id):
        if entailment_cache_id is None:
            return dict()

        logging.info('Restoring prediction cache from %s', entailment_cache_id)

        api = wandb.Api()
        run = api.run(entailment_cache_id)
        run.file(self.entailment_file).download(
            replace=True, exist_ok=False, root=wandb.run.dir)

        with open(f'{wandb.run.dir}/{self.entailment_file}', "rb") as infile:
            return pickle.load(infile)

    def save_prediction_cache(self):
        # write the dictionary to a pickle file
        utils.save(self.prediction_cache, self.entailment_file)

    def check_implication(self, text1, text2, example=None):
        if example is None:
            raise ValueError
        prompt = self.equivalence_prompt(text1, text2, example['question'], prompt_type=self.prompt_type)

        logging.info('%s input: %s', self.name, prompt)

        hashed = oai.md5hash(prompt)
        if hashed in self.prediction_cache:
            logging.info('Restoring hashed instead of predicting with model.')
            response = self.prediction_cache[hashed]
        else:
            if self.entailment_cache_only:
                raise ValueError
            response = self.predict(prompt, temperature=0.02)
            self.prediction_cache[hashed] = response

        logging.info('%s prediction: %s', self.name, response)

        binary_response = response.lower()
        if 'entail' in binary_response:
            return 2
        elif 'neutral' in binary_response:
            return 1
        elif 'contrad' in binary_response:
            return 0
        else:
            logging.warning(f'response, {response}, MANUAL NEUTRAL!')
            return 1

    def batch_check_implication(self, text_list1, text_list2, example=None):
        if example is None:
            raise ValueError
        assert len(text_list1) == len(text_list2)
        full_responses = collections.defaultdict(str)
        batch_prompt, batch_i = [], []
        for i, (text1, text2) in enumerate(zip(text_list1, text_list2)):
            prompt = self.equivalence_prompt(text1, text2, example['question'], prompt_type=self.prompt_type)
            hashed = oai.md5hash(prompt)
            if hashed in self.prediction_cache:
                response = self.prediction_cache[hashed]
                full_responses[i] = response
            else:
                batch_prompt.append(prompt)
                batch_i.append(i)
        if batch_prompt:
            generate_responses = self.batch_predict(batch_prompt, temperature=0.02)
        else:
            generate_responses = []
            
        assert len(generate_responses) == len(batch_prompt) == len(batch_i)
        for i, prompt, response in zip(batch_i, batch_prompt, generate_responses):
            hashed = oai.md5hash(prompt)
            self.prediction_cache[hashed] = response
            full_responses[i] = response
            
        out_list = []
        for i in range(len(text_list1)):
            response = full_responses[i]
            binary_response = response.lower()
            if 'entail' in binary_response:
                out_list.append(2)
            elif 'neutral' in binary_response:
                out_list.append(1)
            elif 'contrad' in binary_response:
                out_list.append(0)
            else:
                logging.warning(f'response, {response}, MANUAL NEUTRAL!')
                out_list.append(1)
        return out_list



class EntailmentVLLM(EntailmentLLM):
    def __init__(self, entailment_cache_id, entailment_cache_only, name, prompt_type='default'):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = 'meta-llama/Llama-3.1-70B-Instruct'
        self.port = name
        self.prompt_type = prompt_type

    def equivalence_prompt(self, text1, text2, question, prompt_type='default'):
        prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2?"""
        if prompt_type == 'ignore_lu':
            prompt += """ Please disregard any expressions of uncertainty such as "may", "might", or "I am uncertain" in the provided answer, and concentrate solely on the content. """
        prompt += " Respond only with entailment, contradiction, or neutral.\nResponse:"""
        return prompt

    def predict(self, prompt, temperature):
        client = openai.OpenAI(
            base_url=self.port,
            api_key="NOT A REAL KEY",
        )
        chat_completion = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user","content": prompt}],
            max_tokens=30,
            temperature=temperature,
        )

        return chat_completion.choices[0].message.content
    
    def batch_predict(self, batch_prompts, temperature):
        all_return_values = thread_map(
            lambda p: self.predict(p, temperature=temperature),
            batch_prompts,
            max_workers=20,
            desc="using vllm")
        return all_return_values

class EntailmentLlama(EntailmentLLM):
    def __init__(self, entailment_cache_id, entailment_cache_only, name, prompt_type='default'):
        super().__init__(entailment_cache_id, entailment_cache_only)
        self.name = name
        self.model = HuggingfaceModel(
            name, stop_sequences='default', max_new_tokens=30)
        self.prompt_type = prompt_type

    def equivalence_prompt(self, text1, text2, question, prompt_type='default'):
        prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2?"""
        if prompt_type == 'ignore_lu':
            prompt += """ Please disregard any expressions of uncertainty such as "may", "might", or "I am uncertain" in the provided answer, and concentrate solely on the content. """
        prompt += " Respond only with entailment, contradiction, or neutral.\nResponse:"""
        return prompt

    def predict(self, prompt, temperature):
        predicted_answer, _, _ = self.model.predict(prompt, temperature)
        return predicted_answer
    
    def batch_predict(self, batch_prompts, temperature):
        all_return_values = self.model.batch_predict(batch_prompts, temperature, num_return_sequences=1, return_full=False, return_latent=False, output_hidden_states=False)
        all_return_values = [x[0] for x in all_return_values]
        return all_return_values


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""
    def are_equivalent(text1, text2):
        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
        return semantically_equivalent

    def are_equivalent_batch(text1, candidates):
        premises = []
        hypotheses = []
        for text2 in candidates:
            premises.append(text1)
            hypotheses.append(text2)
            premises.append(text2)
            hypotheses.append(text1)
        # Batch check implications
        try:
            batch_results = model.batch_check_implication(premises, hypotheses, example=example)
        except Exception as e:
            print(e)
            print("premises", premises)
            print("hypotheses", hypotheses)
            print("text1", text1)
            print("candidates", candidates)
            print("example", example)
            assert False
        # Determine equivalence for each candidate
        equivalent_flags = []
        for idx in range(len(candidates)):
            implication_1 = batch_results[2 * idx]
            implication_2 = batch_results[2 * idx + 1]
            
            if strict_entailment:
                sem_equiv = (implication_1 == 2) and (implication_2 == 2)
            else:
                implications = [implication_1, implication_2]
                sem_equiv = (0 not in implications) and (implications != [1, 1])
            
            equivalent_flags.append(sem_equiv)
        return equivalent_flags


    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    # next_id = 0
    # for i, string1 in enumerate(strings_list):
    #     # Check if string1 already has an id assigned.
    #     if semantic_set_ids[i] == -1:
    #         # If string1 has not been assigned an id, assign it next_id.
    #         semantic_set_ids[i] = next_id
    #         for j in range(i+1, len(strings_list)):
    # if semantic_set_ids[j] == -1:
    #             # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
    #             if are_equivalent(string1, strings_list[j]):
    #                 semantic_set_ids[j] = next_id
    #         next_id += 1

    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            candidates = []
            candidate_indices = []
            for j in range(i + 1, len(strings_list)):
                if semantic_set_ids[j] == -1:
                    candidates.append(strings_list[j])
                    candidate_indices.append(j)
            if candidates:
                equivalent_flags = are_equivalent_batch(string1, candidates)
                for flag, j in zip(equivalent_flags, candidate_indices):
                    if flag:
                        semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    return semantic_set_ids



def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy
