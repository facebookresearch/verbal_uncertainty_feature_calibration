import pickle
import os
cwd = "/private/home/ziweiji/Hallu_Det/"
if not os.path.exists(cwd):
    cwd = "/home/ziweiji/Hallu_Det/"
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoTokenizer,
)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
class HiddenLayersDataset(Dataset):
    def __init__(self, data_paths, split, 
                 layers,
                 save_cache,
                 info_type,
                 label_name='label',
                 ignore_missing_info=False,
                 pair_differ=False,
                model_name='Meta-Llama-3.1-8B-Instruct'):
        super().__init__()
        self.split = split
        self.layers = layers
        self.save_cache = save_cache
        self.ignore_missing_info = ignore_missing_info
        self.pair_differ = pair_differ
        self.info_type = info_type
        self.label_name = label_name
        self.model_name = model_name

        self.LABELMAP = {'hallucinated': 1, "ok": 0}
        if 'only_question' in self.info_type:
            if model_name == 'Meta-Llama-3.1-8B-Instruct':
                self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = []
        for data_path in data_paths:
            data_path = data_path.format(split=split)
            source = "/".join(data_path.split("/")[-3:-1]) #IDK/paired_data_rate_0.5_12000
            data = pd.read_csv(data_path, encoding="utf8")
            data['save_path'] = f'{cwd}/datasets/{source}/{save_cache}/{split}/'
            dataset.append(data)
        self.dataset = pd.concat(dataset, ignore_index=True)
        self.features = self.dataset.columns
        
    def load_hidden_states(self, save_path):
        # t1 = time.time()
        hidden_states_layers = torch.load(save_path, map_location=torch.device('cpu'))
        hidden_states = []
        for layer in self.layers:
            layer = int(layer)
            h = hidden_states_layers[layer]
            if len(h.shape) == 2: # save_cache for each
                if 'only_question' in self.info_type:
                    h = h[:self.question_len]
                if 'last' in self.info_type:
                    h = h[-2]
            # else: # save_cache for last
            hidden_states.append(h)
        hidden_states = torch.stack(hidden_states) # [num_layers, seq_len/1, hidden_dim]
        if self.pair_differ:
            # each two layer is a pair, [i+1] - [i]
            # hidden_states = hidden_states[1::2] - hidden_states[0::2] # [num_layers//2, seq_len, hidden_dim]
            hidden_states = torch.diff(hidden_states, dim=0) # [num_layers-1, seq_len, hidden_dim]
        # t2 = time.time()
        # print('loaded', t2-t1)
        return hidden_states

    def get_question_len(self, row):
        messages = [{"role": "user", "content": row["question"]}] #,{"role": "assistant", "content": row["answer"]}
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        question_len = len(self.tokenizer(prompt, return_tensors='pt')['input_ids'][0])
        return question_len
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # print("index", index, type(index))
        label_name = self.label_name
        row = self.dataset.iloc[index]
        save_path = row['save_path']+f'/{row["id"]}.pt'

        if 'only_question' in self.info_type:
            self.question_len = self.get_question_len(row)

        try:
            hidden_states = self.load_hidden_states(save_path)
            # hidden_states.to(torch.float32)
            if label_name in ['ling_uncertainty',
                               'sentence_semantic_entropy',  'no_refuse_sentence_semantic_entropy', 
                              'word_semantic_entropy', 'no_refuse_word_semantic_entropy', 
                              'word_eigen', 'no_refuse_word_eigen',
                              'sentence_eigen', 'no_refuse_sentence_eigen']: # regression
                l = row[self.label_name]
                l = torch.tensor(l, dtype=torch.float32)
            elif label_name in ['if_refusal']:# classification
                l = row[label_name]
                l = torch.tensor(l, dtype=torch.long)
            else: # classification hallucination
                l = self.LABELMAP[row["label"]]
                l = torch.tensor(l, dtype=torch.long)
        except Exception as e:
            if self.ignore_missing_info:
                print(f"cannot find {row['id']}")
                return None
            else:
                print(f"cannot find {row['id']}")
                print(e)
                print(row["id"])
                assert False

        return {"id":row["id"], 
                "x": hidden_states, 
                "label":l}
    
    def remove_columns(self, columns):
        self.dataset = self.dataset.drop(columns, axis=1)
        return self
    



class HiddenLayers_U_Dataset(HiddenLayersDataset):
    def __init__(self, data_paths, split, 
                 layers,
                 save_cache,
                 info_type,
                 label_name='label',
                 uncertainty_names=[],
                normalize=True,
                use_predicted_uncertainty=False,
                model_name='Meta-Llama-3.1-8B-Instruct'):
        super().__init__(data_paths, split, 
                         layers, save_cache, 
                         info_type, label_name,
                         ignore_missing_info=False, pair_differ=False)
        self.label_name = label_name
        self.uncertainty_names = uncertainty_names
        self.LABELMAP = {"ok": 0, 'hallucinated': 1}
        self.normalize = normalize
        self.use_predicted_uncertainty = use_predicted_uncertainty
        self.model_name = model_name

        assert info_type in ["concat", 'separate', 'only_answer', 'only_question', 'only_answer_plus_uncertainty', 'multi_task']
        if info_type in ["only_answer_plus_uncertainty", 'multi_task']:
            self.normalize = False

        dataset = []
        for data_path in data_paths:
            # /home/ziweiji/Hallu_Det/datasets/trivia_qa/refuse_paired/test.csv
            data_path = data_path.format(split=split)
            dataset_name = data_path.split("/")[-3]
            data = pd.read_csv(data_path, encoding="utf8")
            data = self.get_uncertainty(dataset_name, data)

            data_dir = "/".join(data_path.split("/")[:-1])
            data['activation_path'] = f'{data_dir}/{save_cache}/{split}/'
            data['only_question_activation_path'] = f'{data_dir}/only_question_{save_cache}/{split}/'
            dataset.append(data)

        self.dataset = pd.concat(dataset, ignore_index=True)
        self.features = self.dataset.columns
        
    def get_uncertainty(self, dataset_name, data):
        model_name = self.model_name
        split = self.split
        normalize = self.normalize
        uncertainty_names = self.uncertainty_names
        # if model_name == 'Meta-Llama-3.1-8B-Instruct':
        #     if self.use_predicted_uncertainty:
        #         question_path = f'{cwd}/datasets/{dataset_name}/sampled_predicted_uncertainty/{split}.csv'
        #     else:
        #         question_path = f'{cwd}/datasets/{dataset_name}/sampled/{split}.csv'
        # else:
        if self.use_predicted_uncertainty:
            question_path = f'{cwd}/datasets/{dataset_name}/{model_name}_predicted_uncertainty/{split}.csv'
        else:
            question_path = f'{cwd}/datasets/{dataset_name}/{model_name}/{split}.csv'

        question_data = pd.read_csv(question_path)
        all_uncertainty = []
        for uncertainty_name in uncertainty_names:
            try:
                uncertainties = question_data[uncertainty_name].to_numpy().reshape(-1, 1)
            except:
                print("cannot find uncertainty_name", uncertainty_name)
                print(question_data.columns)
                print("question_path", question_path)
                assert False
            if normalize:
                scaler = MinMaxScaler()
                uncertainties = scaler.fit_transform(uncertainties)
            all_uncertainty.append(uncertainties)
        # shape [num_uncertainty, num_question] to [num_question, num_uncertainty]
        all_uncertainty = list(map(list, zip(*all_uncertainty)))
        question_data['id'] = question_data['id'].astype(str)
        QID2uncertainty = {} # {qid: [uncertainty1, uncertainty2, ...]}
        # print("len(question_data['id']), len(all_uncertainty)", len(question_data['id']), len(all_uncertainty))
        assert len(question_data['id']) == len(all_uncertainty)
        for qid, q_uncertainty in zip(question_data['id'], all_uncertainty):
            QID2uncertainty[qid] = q_uncertainty
            
        # number of question and answer may not equal
        for idx, row in data.iterrows():
            qid = str(self.id2qid(row['id'], dataset_name))
            data.at[idx, 'qid'] = qid
            for u_i, uncertainty_name in enumerate(uncertainty_names):
                if qid not in QID2uncertainty:
                    print(f"cannot find {type(qid)} {qid} in {uncertainty_name}")
                    print(list(QID2uncertainty.keys()))
                    print(type(list(QID2uncertainty.keys())[0]), list(QID2uncertainty.keys())[0])
                    assert False
                data.at[idx, uncertainty_name] = QID2uncertainty[qid][u_i]
        return data

    def id2qid(self, id, dataset_name):
        id = str(id)
        if dataset_name == 'HaluEval':
            qid = id.split('_')[0]
        elif dataset_name == 'IDK':
            qid = id.split('_')[1]
        else:
            qid = id
        return qid

    def load_hidden_states(self, activation_path):
        hidden_states_layers = torch.load(activation_path, map_location=torch.device('cpu'))
        hidden_states = []
        for layer in self.layers:
            layer = int(layer)
            h = hidden_states_layers[layer]
            hidden_states.append(h)
        hidden_states = torch.stack(hidden_states) # [num_layers, seq_len, hidden_dim]
        return hidden_states


    def __getitem__(self, index):
        # print("index", index, type(index))
        uncertainty_names = self.uncertainty_names
        label_name = self.label_name

        row = self.dataset.iloc[index]
        activation_path = row['activation_path']+f'/{row["id"]}.pt'
        only_question_activation_path = row['only_question_activation_path']+f'/{row["qid"]}.pt'
        
        try:
            if 'only_question' in self.info_type:
                hidden_states = None
            else:
                hidden_states = self.load_hidden_states(activation_path)
                
            if 'only_answer' in self.info_type:
                only_question_hidden_states = None
            else:
                only_question_hidden_states = self.load_hidden_states(only_question_activation_path)
            
            l = row[label_name]
            if label_name == 'label':
                l = self.LABELMAP[l]
            l = torch.tensor(l, dtype=torch.long)
            all_uncertainty = []
            for uncertainty_name in uncertainty_names:
                all_uncertainty.append(torch.tensor(row[uncertainty_name], dtype=torch.float32))
            all_uncertainty = torch.stack(all_uncertainty) # [num_uncertainty, 1]
            
        except Exception as e:
            if self.ignore_missing_info:
                print(f"cannot find {row['id']}")
                return None
            else:
                print(f"cannot find {row['id']}")
                print(e)
                print(row["id"])
                assert False

        return {"id": row["id"], 
                "x": hidden_states,
                'question_x': only_question_hidden_states,
                "label":l,
                'all_uncertainty': all_uncertainty}

