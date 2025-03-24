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
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
import logging
import argparse
import pickle

# import wandb
import ast
from evaluate import load
import sys
sys.path.append(current_dir)
from huggingface_models import HuggingfaceModel
import openai as oai
import logging
import hashlib
from tenacity import (retry, stop_after_attempt,  # for exponential backoff
                      wait_random_exponential)

from openai import OpenAI

PROMPTS = {
    'default': "Answer the following question as briefly as possible.\n",
    'chat': 'Answer the following question in a single brief but complete sentence.\n',
    'word': 'Please answer this question. Do not answer in a full sentence. Answer with as few words as possible, e.g. only a name, place, or thing.\n', ## from nature paper
    'sentence': 'Please answer the following question.\n',
    'no_refuse_word': 'Respond to the question with the shortest possible answer, such as a name, place, or thing. Avoid full sentences. If uncertain or unable to answer, still respond directly without additional information, templates, or explanations.\n',
    'no_refuse_sentence': 'Please answer this question. If uncertain or unable to answer, still respond directly without additional information, templates, or explanations.\n'}

def get_parser(stages=['generate', 'compute']):
    entity = os.getenv('WANDB_SEM_UNC_ENTITY', None)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_splits", nargs="*")
    parser.add_argument(
        "--debug", default=False, action='store_true',
        help="Keep default wandb clean.")
    parser.add_argument('--entity', type=str, default='ziweiji')
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument(
        "--metric", type=str, default="squad",
        # choices=['squad', 'llm', 'llm_gpt-3.5', 'llm_gpt-4'],
        help="Metric to assign accuracy to generations.")
    # parser.add_argument(
    #     "--compute_accuracy_at_all_temps",
    #     action=argparse.BooleanOptionalAction, default=True,
    #     help="Compute accuracy at all temperatures or only t<<1.")
    parser.add_argument("--compute_accuracy_at_all_temps", dest='compute_accuracy_at_all_temps', action='store_true')
    parser.add_argument("--no-compute_accuracy_at_all_temps", dest='compute_accuracy_at_all_temps', action='store_false')
    parser.set_defaults(compute_accuracy_at_all_temps=True)
    parser.add_argument(
        "--experiment_lot", type=str, default='Unnamed Experiment',
        help="Keep default wandb clean.")
    if 'generate' in stages:
        parser.add_argument(
            "--model_name", type=str, help="Model name",
        )
        parser.add_argument(
            "--model_max_new_tokens", type=int, default=100,
            help="Max number of tokens generated.",
        )
        parser.add_argument("--output_hidden_states", default=False, action='store_true')
        parser.add_argument("--dataset", type=str, default="triviaqa", help="Dataset to use")
        parser.add_argument(
            "--ood_train_dataset", type=str, default=None,
            choices=['trivia_qa', 'squad', 'bioasq', 'nq', 'svamp'],
            help="Dataset to use to assemble few-shot prompt, p_true prompt, and train p_ik.")
        parser.add_argument(
            "--num_samples", type=int, default=400,
            help="Number of samples to use")
        parser.add_argument("--num_few_shot", type=int, default=0, help="Number of few shot examples to use")
        parser.add_argument(
            "--p_true_num_fewshot", type=int, default=20,
            help="Number of few shot examples to use")
        parser.add_argument(
            "--p_true_hint", default=False, action='store_true',
            help="Get generations for training set?")
        parser.add_argument("--num_generations", type=int, default=10,help="Number of generations to use")
        parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
        parser.add_argument(
            "--use_mc_options", type=bool, default=True,
            help="Include MC options question?")
        parser.add_argument("--get_most_likely_answer", default=False, action='store_true')
        # parser.add_argument(
        #     "--get_training_set_generations", default=True,
        #     action=argparse.BooleanOptionalAction,
        #     help="Get generations for training set?")
        parser.add_argument("--get_training_set_generations", dest='get_training_set_generations', action='store_true')
        parser.add_argument("--no-get_training_set_generations", dest='get_training_set_generations', action='store_false')
        parser.set_defaults(get_training_set_generations=True)
        parser.add_argument("--use_context", default=False, action='store_true', help="Get generations for training set?")
        parser.add_argument("--get_training_set_generations_most_likely_only", dest='get_training_set_generations_most_likely_only', action='store_true')
        parser.add_argument('--compute_p_true', dest='compute_p_true', action='store_true')
        parser.add_argument("--prompt_type", type=str)
        # parser.add_argument(
        #     "--compute_uncertainties", default=True,
        #     action=argparse.BooleanOptionalAction,
        #     help='Trigger compute_uncertainty_measures.py')
        parser.add_argument("--compute_uncertainties", dest='compute_uncertainties', action='store_true')
        parser.add_argument("--no-compute_uncertainties", dest='compute_uncertainties', action='store_false')
        parser.set_defaults(compute_uncertainties=True)
        parser.add_argument("--answerable_only", default=False, action='store_true', help='Exclude unanswerable questions.')

    if 'compute' in stages:
        parser.add_argument('--recompute_accuracy', default=False, action='store_true')
        parser.add_argument('--eval_wandb_runid', type=str,
                            help='wandb run id of the dataset to evaluate on')
        parser.add_argument('--train_wandb_runid', type=str, default=None,
                            help='wandb run id of the dataset from which training embeddings and p_true samples will be taken')
        parser.add_argument('--num_eval_samples', type=int, default=int(1e19))
        # parser.add_argument('--compute_predictive_entropy',
        #                     default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--compute_predictive_entropy', dest='compute_predictive_entropy', action='store_true')
        parser.add_argument('--no-compute_predictive_entropy', dest='compute_predictive_entropy', action='store_false')
        parser.set_defaults(compute_predictive_entropy=True)
        parser.add_argument('--compute_p_ik', dest='compute_p_ik', action='store_true', default=False)
        parser.add_argument('--compute_p_ik_answerable', default=False, action='store_true')
        parser.add_argument('--compute_context_entails_response', default=False, action='store_true')
        # parser.add_argument('--analyze_run', default=True,
        #                     action=argparse.BooleanOptionalAction)
        parser.add_argument('--analyze_run', dest='analyze_run', action='store_true', default=False)
        parser.add_argument('--assign_new_wandb_id', dest='assign_new_wandb_id', action='store_true', default=False)
        parser.add_argument('--restore_entity_eval', type=str, default='ziweiji')
        parser.add_argument('--restore_entity_train', type=str, default='ziweiji')
        # parser.add_argument('--condition_on_question',
        #                     default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--condition_on_question', dest='condition_on_question', action='store_true')
        parser.add_argument('--no-condition_on_question', dest='condition_on_question', action='store_false')
        parser.set_defaults(condition_on_question=True)
        # parser.add_argument('--strict_entailment',
        #                     default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--strict_entailment', dest='strict_entailment', action='store_true')
        parser.add_argument('--no-strict_entailment', dest='strict_entailment', action='store_false')
        parser.set_defaults(strict_entailment=True)
        # parser.add_argument('--use_all_generations', default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_all_generations', dest='use_all_generations', action='store_true')
        parser.add_argument('--no-use_all_generations', dest='use_all_generations', action='store_false')
        parser.set_defaults(use_all_generations=True)
        parser.add_argument('--use_num_generations', type=int, default=-1)
        parser.add_argument("--entailment_model", default='deberta', type=str)
        parser.add_argument("--entailment_cache_id", default=None, type=str, help='Restore entailment predictions from previous run for GPT-4/LLaMa-Entailment.')
        parser.add_argument('--entailment_cache_only', default=False, action='store_true')
        parser.add_argument('--compute_p_true_in_compute_stage',
                            default=False, action='store_true')
        parser.add_argument('--reuse_entailment_model',
                            default=False, action='store_true',
                            help='Use entailment model as p_true model.')
    return parser


def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG


def construct_fewshot_prompt_from_indices(dataset, example_indices, brief, brief_always, make_prompt):
    """Given a dataset and indices, construct a fewshot prompt."""
    if not brief_always:
        prompt = brief
    else:
        prompt = ''

    for example_index in example_indices:

        example = dataset[example_index]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt


def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices


def model_based_metric(predicted_answer, example, model):
    if 'answers' in example:
        correct_answers = example['answers']['text']
    elif 'reference' in example:
        correct_answers = example['reference']['answers']['text']
    else:
        raise ValueError

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"
    predicted_answer = ''
    if 'gpt' in model.model_name.lower():
        predicted_answer = model.predict(prompt, 0.01)
    else:
        predicted_answer, _, _ = model.predict(prompt, 0.01)

    if 'yes' in predicted_answer.lower():
        return 1.0
    elif 'no' in predicted_answer.lower():
        return 0.0
    else:
        logging.warning('Redo llm check.')
        predicted_answer, _, _ = model.predict(prompt, 1)
        if 'yes' in predicted_answer.lower():
            return 1.0
        elif 'no' in predicted_answer.lower():
            return 0.0

        logging.warning('Answer neither no nor yes. Defaulting to no!')
        return 0.0


def llm_metric(predicted_answer, example, model):
    return model_based_metric(predicted_answer, example, model)

def batch_llm_metric(batched_predicted_answer, batched_example, model, prompt_type='default'):
    if len(batched_predicted_answer) != len(batched_example):
        print(f"len(batched_predicted_answer): {len(batched_predicted_answer)}, len(batched_example): {len(batched_example)}")
        print(f"batched_predicted_answer: {batched_predicted_answer}")
        print(f"batched_example: {batched_example}")
    assert len(batched_predicted_answer) == len(batched_example)
    all_prompts = []
    for predicted_answer, example in zip(batched_predicted_answer, batched_example):
        if 'answers' in example:
            correct_answers = example['answers']['text']
        elif 'reference' in example:
            correct_answers = example['reference']['answers']['text']
        elif 'answer' in example:
            if type(example['answer']) == list:
                correct_answers = example['answer']
            else:
                correct_answers = ast.literal_eval(example['answer'])
        else:
            raise ValueError

        prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
        if len(correct_answers) == 1:
            prompt += f"The expected answer is: {correct_answers[0]}.\n"
        else:
            prompt += f"The following are expected answers to this question: {correct_answers}.\n"

        prompt += f"The proposed answer is: {predicted_answer}\n"

        if len(correct_answers) == 1:
            prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
        else:
            prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"
        
        if prompt_type == 'ignore_vu':
            prompt += """ Please disregard any expressions of uncertainty such as "may", "might", or "I am uncertain" in the provided answer, and concentrate solely on the content."""

        prompt += " Respond only with yes or no.\nResponse:"
        all_prompts.append(prompt)
    try:
        batch_res = model.batch_predict(all_prompts, 0.01, output_hidden_states=False)
    except Exception as e:
        print(f"all_prompts: {all_prompts}")
        print(f"batched_predicted_answer: {batched_predicted_answer}")
        print(f"batched_example: {batched_example}")
        raise e
    acces = []
    for predicted_answer in batch_res:
        assert type(predicted_answer) in [list, tuple]
        predicted_answer = predicted_answer[0]
        predicted_answer = predicted_answer.lower().strip()

        if 'yes' in predicted_answer:
            acces.append(1.0)
        elif 'no' in predicted_answer:
            acces.append(0.0)
        else:
            logging.warning('Redo llm check.')
            redo_i = 0
            redo_acc = 0.0
            while redo_i < 5:
                predicted_answer, _, _ = model.predict(prompt, 0.1, output_hidden_states=False)
                predicted_answer = predicted_answer.lower().strip()
                if 'yes' in predicted_answer:
                    redo_acc = 1.0
                    break
                elif 'no' in predicted_answer:
                    break
                else:
                    redo_i += 1
            acces.append(redo_acc)

    if len(acces) != len(batched_predicted_answer):
        print(f"len(acces): {len(acces)}, len(batched_predicted_answer): {len(batched_predicted_answer)}")
        print(f"acces: {acces}")
        print(f"batched_predicted_answer: {batched_predicted_answer}")
    assert len(acces) == len(batched_predicted_answer)
    return acces


def get_gpt_metric(metric_name):

    model_name = '_'.join(metric_name.split('_')[1:])

    class EntailmentGPT():
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, prompt, temperature):
            return oai.predict(prompt, temperature, model=self.model_name)

    gpt_model = EntailmentGPT(model_name)

    def gpt_metric(predicted_answer, example, model):
        del model
        return model_based_metric(predicted_answer, example, gpt_model)

    return gpt_metric


def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    reference = {'answers': {'answer_start': answer_starts, 'text': answers['text']}, 'id': str(example['id'])}
    return reference


def init_model(args):
    mn = args.model_name
    if 'llama' in mn.lower() or 'qwen' in mn.lower() or 'mistral' in mn.lower() or 'phi' in mn.lower():
        model = HuggingfaceModel(
            mn, stop_sequences='default',
            max_new_tokens=args.model_max_new_tokens)
    else:
        raise ValueError(f'Unknown model_name `{mn}`.')
    return model


def make_prompt(prompt_type, question):
    prompt = PROMPTS[prompt_type]
    prompt += f"Question: {question}\nAnswer:"
    return prompt


def get_metric(metric):
    if metric == 'squad':

        squad_metric = load("squad_v2")

        def metric(response, example, *args, **kwargs):
            # Compatibility with recomputation.
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError
            exid = str(exid)
            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}
            results = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])
            return 1.0 if (results['f1'] >= 50.0) else 0.0

    # Reuses the globally active model for these.
    elif metric == 'llm':
        metric = llm_metric
    elif metric == 'llm_gpt-3.5':
        metric = get_gpt_metric(metric)
    elif metric == 'llm_gpt-4':
        metric = get_gpt_metric(metric)
    else:
        raise ValueError

    return metric


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)