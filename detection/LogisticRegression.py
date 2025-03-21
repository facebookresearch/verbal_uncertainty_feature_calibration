import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, f1_score
import argparse
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from importlib import reload
import sys
sys.path.append(f"{root_path}/src/")
from detection_utils import load_data

def train_test(X_train, y_train, X_val, y_val, val_refusal=None, return_all=False):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Create and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    

    # Make predictions
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1]
    if val_refusal:
        y_pred = y_pred[val_refusal]
        y_prob = y_prob[val_refusal]
        y_val = y_val[val_refusal]
    print("y_pred", np.mean(y_pred), len(y_pred))
    print("y_val", np.mean(y_val), len(y_val))


    # Evaluate the model
    # AUROC
    auroc = roc_auc_score(y_val, y_prob, average='macro') *100
    accuracy = accuracy_score(y_val, y_pred) *100
    f1 = f1_score(y_val, y_pred, average='macro') *100
    loss = log_loss(y_val, y_prob)

    precision = precision_score(y_val, y_pred, average='macro') *100
    recall = recall_score(y_val, y_pred, average='macro') *100

    # print(f"Accuracy F1 Pre Recall: {accuracy:.2f}\t{f1:.2f}\t{precision:.2f}\t{recall:.2f}")
    if return_all:
        return auroc, accuracy, f1, precision, recall, model, y_pred, y_prob
    else:
        return auroc, accuracy, f1, precision, recall

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    prompt_type = 'sentence'
    label_name = 'label'
    filter_refusal = False
    use_predicted_test = True
    train_split = 'train'
    model_name = args.model_name
    # model_name = "Qwen2.5-7B-Instruct"
    # model_name = "Mistral-7B-Instruct-v0.3"
    # model_name = 'Meta-Llama-3.1-8B-Instruct'

    outputs = []
    FEATURES = [
        ['sentence_semantic_entropy'],
        ['verbal_uncertainty'],
        ['verbal_uncertainty', f'{prompt_type}_semantic_entropy'],
        ]
        
    for dataset in ['trivia_qa', 'nq_open', 'pop_qa']:
        test_dataset = dataset
        out_dir = f"LR_outputs/{dataset}/{model_name}/"
        os.makedirs(out_dir, exist_ok=True)
        print(f"Training on {dataset}")
        for feature in FEATURES:
            X_train, y_train = load_data(model_name, dataset, train_split, feature, label_name, prompt_type, filter_refusal=filter_refusal)
            X_val, y_val = load_data(model_name, test_dataset, 'test', feature, label_name, prompt_type, use_predicted_test=use_predicted_test, filter_refusal=filter_refusal)
            auroc, accuracy, f1, precision, recall, model, y_pred, y_prob = train_test(X_train, y_train, X_val, y_val, return_all=True)
            auroc = round(auroc, 2)
            accuracy = round(accuracy, 2)
            outputs.append(f"{auroc}\t{accuracy}")
            if use_predicted_test:
                output_path = out_dir+ "_".join(feature)+"_use_predicted_test.json"
            else:
                output_path = out_dir+ "_".join(feature)+".json"

            with open(output_path, 'w') as f:
                data = {"auroc": auroc, "accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall,
                "y_pred": y_pred.tolist(), "y_prob": y_prob.tolist(), "y_val": y_val.tolist()}
                json.dump(data, f)
                
    for o in outputs:
        print(o)