"""Implement HuggingfaceModel models."""
import copy
import logging
import os
from collections import Counter

import accelerate
import torch
from accelerate import Accelerator

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download
import re

from uncertainty.models.base_model import BaseModel
from uncertainty.models.base_model import STOP_SEQUENCES


class StoppingCriteriaSub(StoppingCriteria):
    """Stop generations when they match a particular text or token."""
    def __init__(self, stops, tokenizer, match_on='text', initial_length=None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]
            print(self.stops)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be dangerous due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise
            if match:
                return True
        return False


def remove_split_layer(device_map_in):
    """Modify device maps s.t. individual layers are not spread across devices."""

    device_map = copy.deepcopy(device_map_in)
    destinations = list(device_map.keys())

    counts = Counter(['.'.join(i.split('.')[:2]) for i in destinations])

    found_split = False
    for layer, count in counts.items():
        if count == 1:
            continue

        if found_split:
            # Only triggers if we find more than one split layer!
            raise ValueError(
                'More than one split layer.\n'
                f'Currently at layer {layer}.\n'
                f'In map: {device_map_in}\n'
                f'Out map: {device_map}\n')

        logging.info(f'Split layer is {layer}.')

        # remove split for that layer
        for name in list(device_map.keys()):
            if name.startswith(layer):
                print(f'pop {name}')
                device = device_map.pop(name)

        device_map[layer] = device
        found_split = True

    return device_map


class HuggingfaceModel(BaseModel):
    """HuggingfaceModel."""

    def __init__(self, model_name, stop_sequences=None, max_new_tokens=None):
        if max_new_tokens is None:
            raise
        self.max_new_tokens = max_new_tokens

        # if stop_sequences == 'default':
        #     stop_sequences = STOP_SEQUENCES
        print(model_name)
        if 'llama' in model_name.lower():

            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
                eightbit = True
            else:
                kwargs = {}
                eightbit = False

            if 'Llama-2' in model_name or 'Llama-3' in model_name:
                base = 'meta-llama'
            self.tokenizer = AutoTokenizer.from_pretrained(
                f"{base}/{model_name}", device_map="auto",
                token_type_ids=None, padding_side='left')

            self.model = AutoModelForCausalLM.from_pretrained(f"{base}/{model_name}", device_map="auto", **kwargs,)
            # if ('7b' in model_name or '13b' in model_name) or eightbit:
            #     self.model = AutoModelForCausalLM.from_pretrained(
            #         f"{base}/{model_name}", device_map="auto",
            #         max_memory={0: '80GIB'}, **kwargs,)

            # elif llama2or3_70b or llama65b:
            #     path = snapshot_download(
            #         repo_id=f'{base}/{model_name}',
            #         allow_patterns=['*.json', '*.model', '*.safetensors'],
            #         ignore_patterns=['pytorch_model.bin.index.json']
            #     )
            #     config = AutoConfig.from_pretrained(f"{base}/{model_name}")
            #     with accelerate.init_empty_weights():
            #         self.model = AutoModelForCausalLM.from_config(config)
            #     self.model.tie_weights()
            #     if 'chat' in model_name:
            #         max_mem = 17.5 * 4686198491
            #     else:
            #         max_mem = 15 * 4686198491
                
            #     device_map = accelerate.infer_auto_device_map(
            #         self.model.model,
            #         max_memory={0: max_mem, 1: max_mem},
            #         dtype='float16'
            #     )
            #     device_map = remove_split_layer(device_map)
            #     full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
            #     full_model_device_map["lm_head"] = 0

            #     self.model = accelerate.load_checkpoint_and_dispatch(
            #         self.model, path, device_map=full_model_device_map,
            #         dtype='float16', skip_keys='past_key_values')

            # else:
            #     raise ValueError
        elif 'mistral' in model_name.lower():
            if model_name.endswith('-8bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_8bit=True,)}
                model_name = model_name[:-len('-8bit')]
            if model_name.endswith('-4bit'):
                kwargs = {'quantization_config': BitsAndBytesConfig(
                    load_in_4bit=True,)}
                model_name = model_name[:-len('-8bit')]
            else:
                kwargs = {}

            model_id = f'mistralai/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map="auto",
                token_type_ids=None, padding_side='left')

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map='auto',
                max_memory={0: '80GIB'},
                **kwargs,
            )
        elif 'qwen' in model_name.lower():
            model_id = f'Qwen/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map="auto",
                token_type_ids=None, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
            )

        elif 'falcon' in model_name:
            model_id = f'tiiuae/{model_name}'
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)

            kwargs = {'quantization_config': BitsAndBytesConfig(
                load_in_8bit=True,)}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                **kwargs,
            )
        elif 'phi' in model_name.lower():
            model_id = f'microsoft/{model_name}'  # e.g. Phi-3-mini-128k-instruct
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
            )
        elif 'gemma' in model_name:
            model_id = f'google/{model_name}'  # e.g. gemma-7b-it
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, device_map='auto', token_type_ids=None,
                clean_up_tokenization_spaces=False)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map='auto',
                torch_dtype=torch.bfloat16
            )
        else:
            raise ValueError

        self.model_name = model_name
        self.stop_sequences = [self.tokenizer.eos_token]
        self.token_limit = 4096 #if 'Llama-2' in model_name else 2048

    
    def predict(self, input_data, temperature,
                return_full=False, return_latent=False, output_hidden_states=True):
        
        if 'instruct' in self.model_name.lower():
            input_data = self.tokenizer.apply_chat_template([{"role": "user", "content": input_data},], tokenize=False)
            
        inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")
        pad_token_id = self.tokenizer.eos_token_id

        initial_length = len(inputs['input_ids'][0])
        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=initial_length,
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None
        # https://huggingface.co/docs/transformers/v4.47.1/en/main_classes/text_generation#transformers.GenerationConfig
        # model.generation_config
        # top_p = 0.9 and top_k = 50
        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=output_hidden_states,
                top_p=0.9,
                top_k=50,
                temperature=temperature,
                do_sample=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
            )

        if len(outputs.sequences[0]) > self.token_limit:
            print(
                'Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)

        full_answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        full_answer = re.sub("<\|start_header_id\|>assistant<\|end_header_id\|>\n\n", "", full_answer)
        full_answer = re.sub("<\|eot_id\|>", "", full_answer)
        
        # Remove input from answer.
        answer = self.tokenizer.decode(outputs.sequences[0][initial_length:], skip_special_tokens=False)
        n_generated, sliced_answer, token_stop_index = self.get_n_generated(answer)
        # Get log_likelihoods.
        transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        log_likelihoods = self.get_log_likelihoods(transition_scores, 0, n_generated)

        if output_hidden_states:
            if 'decoder_hidden_states' in outputs.keys():
                hidden = outputs.decoder_hidden_states
            else:
                hidden = outputs.hidden_states
                # the first dim is the number of generated token
                # the second dim is the number of layers
                # hidden[0][0] = [batch_size, input_length, hidden_size]
                # hidden[other token][0] = [batch_size, 1, hidden_size]
            hidden_states = self.get_hidden_states(hidden, n_generated, return_latent, 0,
                                                full_answer, sliced_answer, initial_length, token_stop_index)
        else:
            hidden_states = (None, None, None)
        return_values = (sliced_answer, log_likelihoods, hidden_states)
        return return_values

    
    def get_n_generated(self, answer):
        # Remove stop_words from answer.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                index = answer.find(stop)
                if index != -1:
                    stop_at = min(stop_at, index)
            sliced_answer = answer[:stop_at]

            if not all([stop not in sliced_answer for stop in self.stop_sequences]):
                error_msg = 'Error: Stop words not removed successfully!'
                error_msg += f'Answer: >{answer}< '
                error_msg += f'Sliced Answer: >{sliced_answer}<'
                logging.error(error_msg)

        # Remove whitespaces from answer (in particular from beginning.)
        sliced_answer = sliced_answer.strip()
        token_stop_index = self.tokenizer(answer[:stop_at], return_tensors="pt")['input_ids'].shape[1]
        n_generated = token_stop_index

        if n_generated == 0:
            logging.warning('Only stop_words were generated. For likelihoods and embeddings, taking stop word instead.')
            n_generated = 1
        sliced_answer = re.sub("<\|start_header_id\|>assistant<\|end_header_id\|>\n\n", "", sliced_answer)
        return n_generated, sliced_answer, token_stop_index

    def get_log_likelihoods(self, transition_scores, batch_i, n_generated):
        log_likelihoods = [score.item() for score in transition_scores[batch_i]]
        if len(log_likelihoods) == 1:
            logging.warning('Taking first and only generation for log likelihood!')
            log_likelihoods = log_likelihoods
        else:
            log_likelihoods = log_likelihoods[:n_generated]
        if len(log_likelihoods) == self.max_new_tokens:
            logging.warning('Generation interrupted by max_token limit.')
        if len(log_likelihoods) == 0:
            raise ValueError
        return log_likelihoods
    
    def get_hidden_states(self, hidden, n_generated, return_latent, batch_i,
                          full_answer=None, sliced_answer=None, n_input_token=None, token_stop_index=None):
        if len(hidden) == 1:
            logging.warning(
                'Taking first and only generation for hidden! '
                'n_generated: %d, n_input_token: %d, token_stop_index %d, '
                'generation was: %s',
                n_generated, n_input_token, token_stop_index,
                full_answer,
                )
            last_input = hidden[0]
        elif ((n_generated - 1) >= len(hidden)):
            # if access idx is larger/equal
            logging.error(
                'Taking last state because n_generated is too large'
                'n_generated: %d,  token_stop_index %d, '
                'generation was: %s, slice_answer: %s',
                n_generated,  token_stop_index,
                full_answer, sliced_answer
                )
            last_input = hidden[-1]
        else:
            if len(hidden) != n_generated:
                logging.warning('n_generated: %d, len(hidden): %d', n_generated, len(hidden))
            last_input = hidden[n_generated - 1]
        # last_input is the last generated token
        # Then access last layer for input.
        last_layer = last_input[-1] # shape is [batch_size, 1, hidden_size]
        # Then access last token in input.
        last_token_embedding = last_layer[batch_i, -1, :].cpu()

        if return_latent:
            # Stack second last token embeddings from all layers 
            if len(hidden) == 1:  # FIX: runtime error for mistral-7b on bioasq
                sec_last_input = hidden[0]
            elif ((n_generated - 2) >= len(hidden)):
                sec_last_input = hidden[-2]
            else:
                sec_last_input = hidden[n_generated - 2] # the second last generated token
            sec_last_token_embedding = torch.stack([layer[batch_i, -1, :] for layer in sec_last_input]).cpu()
    
            # Get the last input token embeddings (before generated tokens)
            last_tok_bef_gen_input = hidden[0]
            # shape is [batch_size, initial_length!!!!, hidden_size]
            last_tok_bef_gen_embedding = torch.stack([layer[batch_i, -1, :] for layer in last_tok_bef_gen_input]).cpu()
            hidden_states = (last_token_embedding, sec_last_token_embedding, last_tok_bef_gen_embedding)
        else:
            hidden_states = (last_token_embedding, None, None)
        return hidden_states

    def batch_predict(self, batch_input_data, temperature, num_return_sequences=1,
                      return_full=False, return_latent=False, output_hidden_states=True):
        assert batch_input_data
        # if 'instruct' in self.model_name.lower():
        batch_input_data2 = []
        for input_data in batch_input_data:
            input_data = self.tokenizer.apply_chat_template([{"role": "user", "content": input_data},], tokenize=False)
            batch_input_data2.append(input_data)
        
        if 'llama' in self.model_name.lower() or 'falcon' in self.model_name or 'mistral' in self.model_name.lower():
            # if 'token_type_ids' in inputs:  # HF models seems has changed.
            #     del inputs['token_type_ids']
            pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token_id = pad_token_id
        else:
            pad_token_id = None
        assert self.tokenizer.padding_side == 'left'
        try:
            inputs = self.tokenizer(batch_input_data2, return_tensors="pt", padding=True, truncation=True).to("cuda")
        except Exception as e:
            print(e)
            print("batch_input_data", batch_input_data, batch_input_data2)
            assert False

        initial_length = inputs['input_ids'].shape[1]
        # if self.stop_sequences is not None:
        #     stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
        #         stops=self.stop_sequences,
        #         initial_length=initial_length,
        #         tokenizer=self.tokenizer)])
        # else:
        #     stopping_criteria = None

        logging.debug('temperature: %f', temperature)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=output_hidden_states,
                top_p=0.9,
                top_k=50,
                temperature=temperature,
                do_sample=True,
                # stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                num_return_sequences=num_return_sequences,
            )

        if len(outputs.sequences[0]) > self.token_limit:
            print('Generation exceeding token limit %d > %d',
                len(outputs.sequences[0]), self.token_limit)
        
        batch_answer = self.tokenizer.batch_decode(outputs.sequences[:, initial_length:], skip_special_tokens=False)
        assert len(batch_answer) == len(batch_input_data)*num_return_sequences
        # Get log_likelihoods. [batch_size, n_generated]
        transition_scores = self.model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        assert len(transition_scores) == len(batch_input_data)*num_return_sequences

        if output_hidden_states:
            hidden = outputs.hidden_states
        all_return_values = []
        for batch_i, answer in enumerate(batch_answer):
            n_generated, sliced_answer, token_stop_index = self.get_n_generated(answer)
            log_likelihoods = self.get_log_likelihoods(transition_scores, batch_i, n_generated)
            if output_hidden_states:
                hidden_states = self.get_hidden_states(hidden, n_generated, return_latent, batch_i,
                                               answer, sliced_answer, initial_length, token_stop_index)
            else:
                hidden_states = (None, None, None)
            all_return_values.append((sliced_answer, log_likelihoods, hidden_states))
        assert len(all_return_values) == len(batch_input_data)*num_return_sequences
        return all_return_values

    def get_p_true(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        input_data += ' A'
        tokenized_prompt_true = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        target_ids_true = tokenized_prompt_true.clone()
        # Set all target_ids except the last one to -100.
        target_ids_true[0, :-1] = -100

        with torch.no_grad():
            model_output_true = self.model(tokenized_prompt_true, labels=target_ids_true)

        loss_true = model_output_true.loss

        return -loss_true.item()

    def get_perplexity(self, input_data):
        """Get the probability of the model anwering A (True) for the given input"""

        tokenized_data = self.tokenizer(input_data, return_tensors='pt').to('cuda')['input_ids']

        with torch.no_grad():
            model_output_true = self.model(tokenized_data, labels=tokenized_data)

        perplexity = - model_output_true.loss.item()


        return perplexity
