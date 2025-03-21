
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  f1_score, accuracy_score, log_loss, precision_score, recall_score
import pandas as pd
import json
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)

# 查找最佳阈值
def get_threshold(thresholds, tpr, fpr):
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    return thresholdOpt

def sigmoid(x, k):
    return 1 / (1 + np.exp(-k * x))

def calculate_loss(Label, Score, thresh):
    k = 1
    # sigmoid function
    difference = np.array(Score) - thresh
    p_class1 = sigmoid(difference, k)
    p_class0 = 1 - p_class1
    proba = np.column_stack((p_class0, p_class1))

    loss = log_loss(Label, proba)
    return loss

def getAccuracyF1(Label, Score, thresh, average):
    predict = []
    for ind, item in enumerate(Score):
        predict.append(int(item>=thresh))
    loss = calculate_loss(Label, Score, thresh)

    acc = accuracy_score(Label, predict)
    f1 = f1_score(Label, predict, average=average)
    pre = precision_score(Label, predict, average=average)
    recall = recall_score(Label, predict, average=average)
    return acc, f1, pre, recall



def load_data(model_name, dataset, split, feature, label_name, prompt_type, use_predicted_test=False, filter_refusal=False):
    if dataset == 'IDK':
        assert 'sentence' in prompt_type
        input_file = f"{root_path}/datasets/IDK/paired_data_rate_0.43_12000/{split}.csv"
    else:
        input_file = f'{root_path}/datasets/{dataset}/{model_name}_{prompt_type}/{split}.csv' 
    data = pd.read_csv(input_file)

    if label_name == 'label':
        label = data['label']
        # transform label to binary
        label = label.apply(lambda x: 1 if x == 'hallucinated' else 0)
    elif label_name == 'accuracy':
        # Label = [1 if x == 0.0 else 0 for x in Label]
        label = data['accuracy']
        label = label.apply(lambda x: 1 if x < 1.0 else 0)
    else:
        raise ValueError("label_name should be either 'label' or 'accuracy'")
    
    
    if split == 'train':
        print("label hallucinated rate", np.mean(label))
    
    if use_predicted_test and split == 'test':
        QID2feature = load_predicted_test_data(model_name, dataset, feature)
    else:
        input_file2 = f"{root_path}/datasets/{dataset}/{model_name}/{split}.csv"
        data2 = pd.read_csv(input_file2)
        QID2feature = {}
        for i, row in data2.iterrows():
            try:
                QID2feature[str(row['id'])] = row[feature]
            except:
                print('feature', feature)
                print(row)
                assert False
        
    
    features = []
    for i, row in data.iterrows():
        if dataset == 'IDK':
            qid = row['id'].split('_')[-2]
        else:
            qid = str(row['id'])
        features.append(QID2feature[qid])
    features = pd.DataFrame(features)

    refusal = (data['label'] == 'ok') & (data['accuracy'] == 0)
    if filter_refusal:
        # print(np.mean(refusal))
        label = label[~refusal]
        features = features[~refusal]
    features = features.to_numpy()
    assert len(features) == len(label)
    return features, label

LLAMA_PROBE_PATHS = {
    # "ling_uncertainty": {
    #     "trivia_qa": "outputs/LinearRegressor_ling_uncertainty/trivia_qa_sampled/0.01_range(11,13)",
    #     "nq_open": "outputs/LinearRegressor_ling_uncertainty/nq_open_sampled/0.005_range(11,13)",
    #     "pop_qa": "outputs/LinearRegressor_ling_uncertainty/pop_qa_sampled/0.005_range(11,13)",
    # },
    # "ling_uncertainty": {
    #     "trivia_qa": "outputs/LinearRegressor_ling_uncertainty/trivia_qa_sampled/0.0005_range(5,20)",
    #     "nq_open": "outputs/LinearRegressor_ling_uncertainty/nq_open_sampled/0.001_range(10,20)",
    #     "pop_qa": "outputs/LinearRegressor_ling_uncertainty/pop_qa_sampled/0.001_range(5,20)",
    # },
    "ling_uncertainty": {
        "trivia_qa": "outputs/LinearRegressor_ling_uncertainty/trivia_qa_Meta-Llama-3.1-8B-Instruct/0.001_range(5,20)",
        "nq_open": "outputs/LinearRegressor_ling_uncertainty/nq_open_Meta-Llama-3.1-8B-Instruct/0.001_range(10,20)",
        "pop_qa": "outputs/LinearRegressor_ling_uncertainty/pop_qa_Meta-Llama-3.1-8B-Instruct/0.001_range(10,20)",
    },
    "word_semantic_entropy": {
        "trivia_qa": "outputs/LinearRegressor_word_semantic_entropy/trivia_qa_sampled/0.005_range(12,14)",
        "nq_open": "outputs/LinearRegressor_word_semantic_entropy/nq_open_sampled/0.01_range(12,14)",
        "pop_qa": "outputs/LinearRegressor_word_semantic_entropy/pop_qa_sampled/0.01_range(12,14)",
    },
    "word_eigen": {
        "trivia_qa": "outputs/LinearRegressor_word_eigen/trivia_qa_sampled/0.01_range(12,14)",
        "nq_open": "outputs/LinearRegressor_word_eigen/nq_open_sampled/0.005_range(12,14)",
        "pop_qa": "outputs/LinearRegressor_word_eigen/pop_qa_sampled/0.05_range(12,14)",
    },
    # "sentence_semantic_entropy": {
    #     "trivia_qa": "outputs/LinearRegressor_sentence_semantic_entropy/trivia_qa_sampled/0.005_range(12,14)",
    #     "nq_open": "outputs/LinearRegressor_sentence_semantic_entropy/nq_open_sampled/0.01_range(12,14)",
    #     "pop_qa": "outputs/LinearRegressor_sentence_semantic_entropy/pop_qa_sampled/0.005_range(12,14)",
    # },
    "sentence_semantic_entropy": {
        "trivia_qa": "outputs/LinearRegressor_sentence_semantic_entropy/trivia_qa_Meta-Llama-3.1-8B-Instruct/0.0005_range(10,20)",
        "nq_open": "outputs/LinearRegressor_sentence_semantic_entropy/nq_open_Meta-Llama-3.1-8B-Instruct/0.001_range(10,20)",
        "pop_qa": "outputs/LinearRegressor_sentence_semantic_entropy/pop_qa_Meta-Llama-3.1-8B-Instruct/0.001_range(5,25)",
    },
    "sentence_eigen": {
        "trivia_qa": "outputs/LinearRegressor_sentence_eigen/trivia_qa_sampled/0.005_range(12,14)",
        "nq_open": "outputs/LinearRegressor_sentence_eigen/nq_open_sampled/0.01_range(12,14)",
        "pop_qa": "outputs/LinearRegressor_sentence_eigen/pop_qa_sampled/0.05_range(12,14)",
    },
    }

MISTRAL_PROBE_PATHS ={
    "ling_uncertainty": {
        "trivia_qa": "outputs/LinearRegressor_ling_uncertainty/trivia_qa_Mistral-7B-Instruct-v0.3/0.0005_range(5,20)",
        "nq_open": "outputs/LinearRegressor_ling_uncertainty/nq_open_Mistral-7B-Instruct-v0.3/0.001_range(5,20)",
        "pop_qa": "outputs/LinearRegressor_ling_uncertainty/pop_qa_Mistral-7B-Instruct-v0.3/0.005_range(5,20)",
    },
    "sentence_semantic_entropy": {
        "trivia_qa": "outputs/LinearRegressor_sentence_semantic_entropy/trivia_qa_Mistral-7B-Instruct-v0.3/0.001_range(5,20)",
        "nq_open": "outputs/LinearRegressor_sentence_semantic_entropy/nq_open_Mistral-7B-Instruct-v0.3/0.001_range(5,20)",
        "pop_qa": "outputs/LinearRegressor_sentence_semantic_entropy/pop_qa_Mistral-7B-Instruct-v0.3/0.005_range(5,20)",
    },

}

QWEN_PROBE_PATHS ={
    "ling_uncertainty": {
        "trivia_qa": "outputs/LinearRegressor_ling_uncertainty/trivia_qa_Qwen2.5-7B-Instruct/5e-05_range(5,20)",
        "nq_open": "outputs/LinearRegressor_ling_uncertainty/nq_open_Qwen2.5-7B-Instruct/0.0001_range(10,20)",
        "pop_qa": "outputs/LinearRegressor_ling_uncertainty/pop_qa_Qwen2.5-7B-Instruct/0.001_range(10,20)",
    },
    "sentence_semantic_entropy":{
        "trivia_qa": "outputs/LinearRegressor_sentence_semantic_entropy/trivia_qa_Qwen2.5-7B-Instruct/0.001_range(10,20)",
        "nq_open": "outputs/LinearRegressor_sentence_semantic_entropy/nq_open_Qwen2.5-7B-Instruct/0.0005_range(5,15)",
        "pop_qa": "outputs/LinearRegressor_sentence_semantic_entropy/pop_qa_Qwen2.5-7B-Instruct/5e-05_range(5,20)",
    },

}
def load_predicted_test_data(model_name, dataset, features):
    if model_name == 'Meta-Llama-3.1-8B-Instruct':
        PROBE_PATHS = LLAMA_PROBE_PATHS
    elif model_name == 'Mistral-7B-Instruct-v0.3':
        PROBE_PATHS = MISTRAL_PROBE_PATHS
    elif model_name == 'Qwen2.5-7B-Instruct':
        PROBE_PATHS = QWEN_PROBE_PATHS
    all_features_preds = []
    for f in features:
        input_file2 = f"{root_path}/probe/"+PROBE_PATHS[f][dataset]+f"/{dataset}_predict_results.json"
        with open(input_file2) as f:
            data = json.load(f)
            predictions = data["predictions"]
            mse = data["mse"]['mse']
            print("mse", mse, input_file2)
            all_features_preds.append(predictions)
            
    all_features_preds = np.array(all_features_preds).T

    input_file2 = f"{root_path}/datasets/{dataset}/{model_name}/test.csv"
    
    data2 = pd.read_csv(input_file2)
    assert len(data2) == len(all_features_preds)
    QID2feature = {}
    for i, row in data2.iterrows():
        QID2feature[str(row['id'])] = all_features_preds[i]
    return QID2feature
