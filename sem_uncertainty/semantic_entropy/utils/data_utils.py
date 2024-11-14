"""Data Loading Utilities."""
import logging
import os
import json
import hashlib
import datasets
import ast

def load_ds(dataset_name, seed, add_options=None):
    cache_dir='/home/ziweiji/Hallu_Det/sem_uncertainty/semantic_entropy/cache'
    """Load dataset."""
    if dataset_name == "pop_qa":
        data_files = {'train': '/home/ziweiji/Hallu_Det/datasets/pop_qa/sampled/train.csv', 
                        'val': '/home/ziweiji/Hallu_Det/datasets/pop_qa/sampled/val.csv',
                        'test': '/home/ziweiji/Hallu_Det/datasets/pop_qa/sampled/test.csv'}
        dataset = datasets.load_dataset('csv', data_files=data_files, cache_dir=cache_dir)
        
        # id,subj,prop,obj,subj_id,prop_id,obj_id,s_aliases,o_aliases,s_uri,o_uri,s_wiki_title,o_wiki_title,s_pop,o_pop,question,possible_answers
        dataset = dataset.map(lambda example: {
            'id': str(example['id']),
            'question': example['question'].strip(),
            'answers': {'text': ast.literal_eval(example['answer'])},
            'context': '',
        })

    elif dataset_name == 'nq_open':
        data_files = {'train': '/home/ziweiji/Hallu_Det/datasets/nq_open/sampled/train.csv', 
                        'val': '/home/ziweiji/Hallu_Det/datasets/nq_open/sampled/val.csv',
                        'test': '/home/ziweiji/Hallu_Det/datasets/nq_open/sampled/test.csv'}
        dataset = datasets.load_dataset('csv', data_files=data_files, cache_dir=cache_dir)
        

        dataset = dataset.map(lambda example: {
            'id': str(example['id']),
            'question': example['question'].strip()+'?',
            'answers': {'text': ast.literal_eval(example['answer'])},
            'context': '',
        })
        

    elif dataset_name == "trivia_qa":
        print('load trivia_qa dataset')
        data_files = {'train': '/home/ziweiji/Hallu_Det/datasets/trivia_qa/sampled/train.csv', 
                        'val': '/home/ziweiji/Hallu_Det/datasets/trivia_qa/sampled/val.csv',
                        'test': '/home/ziweiji/Hallu_Det/datasets/trivia_qa/sampled/test.csv'}
        dataset = datasets.load_dataset('csv', data_files=data_files, cache_dir=cache_dir)
        dataset = dataset.map(lambda example: {'id': example['id'], 
                                               'question': example['question'], 
                                                'context': "", 
                                                'answers':{'text': ast.literal_eval(example['answer'])}})
        
    

    columns_to_remove = [col for col in list(dataset.values())[0].column_names if col not in ['question', 'answers', 'id', 'context']]
    print("columns_to_remove", columns_to_remove)
    dataset = dataset.remove_columns(columns_to_remove)
    train_dataset = dataset['train']
    validation_dataset = dataset['val']
    test_dataset = dataset['test']
    return train_dataset, validation_dataset, test_dataset
