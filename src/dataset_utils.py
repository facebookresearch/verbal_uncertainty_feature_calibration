import re
import os
import jsonlines
import numpy as np
import random
import csv


class HightlightedDoc:
    def __init__(self):
        DIT2FILE = {}
        dir = "/home/zjiad/Hallucination_corpus/4AnswerAnnotation/automatic/res_2023_10/"
        for source in os.listdir(dir):
            if "person3" in source:
                continue
            if os.path.isdir(dir+source):
                path = None
                for file in os.listdir(dir+source):
                    if re.match(rf'^{source}_\d+\.jsonl$', file):
                        path = f"{dir+source}/{file}"
                        break
                if not path:
                    if source == "person1":
                        path = f"{dir+source}/output_person1_text2vec_bge_1000.jsonl"
                    elif source == "person4":
                        path = f"{dir+source}/person4_540_3240.jsonl"
                    else:
                        print(f"no file for {source}")
                        assert False

                with jsonlines.open(path) as f:
                    for line in f:
                        DIT2FILE[line["name"]] = path
        self.DIT2FILE = DIT2FILE

    def get_text(self, name):
        file = self.DIT2FILE[name]
        with jsonlines.open(file) as f:
            for line in f:
                if line["name"] == name:
                    return line["all_highlighted_doc_InternLM"], line['selected_questions']
        print('cannot find', name)


   
def sentence_tokenize_process_dot(text, recover=False):
    if not recover:
        text = re.sub(r"O\.S\.B\.M. ", r"O.S.B.M.", text)
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z]\.) ?([A-Za-z])", r"\1\2\3\4", text)
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Za-z])", r"\1\2\3", text)  # J. K. Campbell
        text = re.sub(r"((\n\s*)|(\. ))(\d+)\.\s+", r"\1\4.", text) #1. XXX
        text = re.sub(r"^(\d+)\.\s+", r"\1.", text) #1. XXX
        text = re.sub(r"(\W|^)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec|No|Op|D|Dr|St)\.\s+", r"\1\2.", text)
        text = re.sub(r"(\W|^)(et al)\.\s+([a-z])", r"\1\2.\3", text)
        text = re.sub(r"Alexander v\. Holmes", r"Alexander v.Holmes", text)
        text = re.sub(r"Brown v\. Board", r"Brown v.Board", text)
    else:
        text = re.sub(r"^(\d+)\.", r"\1. ", text) #1. XXX
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z]\.) ?([A-Za-z])", r"\1\2 \3 \4", text) # J. K. Campbell
        text = re.sub(r"(\W|^)([A-Z]\.) ?([A-Z][a-z])", r"\1\2 \3", text)  # J. Campbell
        text = re.sub(r"(\W|^)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec|No|Op|D|Dr|St)\.", r"\1\2. ", text)
        text = re.sub(r"(\W|^)(et al)\.([a-z])", r"\1\2. \3", text)
        
        text = re.sub("O\.S\.B\.M\.", "O.S.B.M. ", text)
        text = re.sub("U\. +S\.", "U.S.", text)
        text = re.sub("U\.S\. *S\. *R\.", "U.S.S.R.", text)
        text = re.sub("D\. +C\.", "D.C.", text)
        text = re.sub("D\. +Roosevelt", "D. Roosevelt", text)
        text = re.sub("A\. *D\. *(\W)", r"A.D.\1", text)
        text = re.sub("A\. +D\.", r"A.D.", text)
        text = re.sub("F\. +C\.", r"F.C.", text)
        text = re.sub("J\. +League", r"J.League", text)
        text = re.sub(r"Alexander v\. *Holmes", r"Alexander v. Holmes", text)
        text = re.sub(r"Brown v\. *Board", r"Brown v. Board", text)
    return text
    
def sentence_tokenize(text, language, keep_end, keep_colon=False):
    if language == 'zh':
        if not keep_colon:
            text = re.sub(r"([:：])(\s+)", r"\1", text)
        sents2 = []
        sents = re.split("(。|！|？|；|\n+)", text) 
        # print(sents)
        for i in range(0, len(sents), 2):
            if i+1<len(sents):
                sent = sents[i] + sents[i+1]
                if not keep_end:
                    sent = sent.strip()
            else:
                sent = sents[i]
                if not keep_end:
                    sent = sent.strip()
            if sent:
                sents2.append(sent)
        # print(sents2)
        return sents2  
    elif language == 'en':
        text = sentence_tokenize_process_dot(text)
        if not keep_colon: #放在处理1.之后
            text = re.sub(r"([:：])(\s+)", r"\1 ", text) #比中文多空格
        
        sents2 = []
        sents = re.split("((?:[.!?;]\s+)|(?:\n+))", text)
        # print(sents)
        for i in range(0, len(sents), 2):
            if i+1<len(sents):
                sent = sents[i] + sents[i+1]
                if not keep_end:
                    sent = sent.strip()
            else:
                sent = sents[i]
                if not keep_end:
                    sent = sent.strip()
            if sent:
                sent = sentence_tokenize_process_dot(sent, recover=True)
                sents2.append(sent)
        return sents2 

def half_sentence_tokenize(text, language, keep_end):
    if language == 'zh':
        sents2 = []
        sents = re.split("([。！？；\n，：:])", text) 
        # print(sents)
        for i in range(0, len(sents), 2):
            if i+1<len(sents):
                sent = sents[i] + sents[i+1]
                if not keep_end:
                    sent = sent.strip()
            else:
                sent = sents[i]
                if not keep_end:
                    sent = sent.strip()
            if sent:
                sents2.append(sent)
        # print(sents2)
    
    else:
        text = sentence_tokenize_process_dot(text)
        sents2 = []
        sents = re.split("([.!?;\n,:] )", text) 
        # print(sents)
        for i in range(0, len(sents), 2):
            if i+1<len(sents):
                sent = sents[i] + sents[i+1]
                if not keep_end:
                    sent = sent.strip()
            else:
                sent = sents[i]
                if not keep_end:
                    sent = sent.strip()
            if sent:
                sent = sentence_tokenize_process_dot(sent, recover=True)
                sents2.append(sent)
        # print(sents2)
    return sents2

    
def cut(text, max_len, language, encoding, keep_end):
    if len(encoding.encode(text)) <= max_len:
        return text
    
    current = ""
    if language == "zh" or keep_end:
        for sent in sentence_tokenize(text, language, keep_end):
            if len(encoding.encode(current+sent)) >= max_len:
                return current
            else:
                current += sent
    elif language == "en":
        for sent in sentence_tokenize(text, language, keep_end):
            if len(encoding.encode(current+" "+sent)) >= max_len:
                return current
            else:
                current += " "+sent
    else:
        assert False
    return current
            
        
def include_hallucination(anns):
    anns2 = []
    for sent in anns:
        if re.search("<Hallucination> (Unverifiable)|(Contradictory)", sent):
            anns2.append(False)
        else:
            anns2.append(True)
    return anns2


def split_subset(all_samples, train_ratio=0.8):
    # all_samples follows the sequence of answers
    all_labels = []
    for s in all_samples:
        if s['label'] == 'ok':
            all_labels.append(1)
        else:
            all_labels.append(0)
    total_sentence_num = len(all_samples)
    ok_rate = np.mean(all_labels)
    print("total_sentence_num", total_sentence_num, "ok_rate", ok_rate)

    train_size = int(train_ratio * total_sentence_num)
    ok_train_size = int(train_size * ok_rate)
    hall_train_size = train_size - ok_train_size
    print("train", train_size, ok_train_size, hall_train_size)

    val_ratio = (1-train_ratio) / 2
    val_size = int(val_ratio * total_sentence_num)
    ok_val_size = int(val_size * ok_rate)
    hall_val_size = val_size - ok_val_size
    print("val", val_size, ok_val_size, hall_val_size)

    test_size = total_sentence_num - train_size - val_size
    ok_test_size = int(test_size * ok_rate)
    hall_test_size = test_size - ok_test_size
    print('test', test_size, ok_test_size, hall_test_size)


    train_ok, train_hall = [], []
    val_ok, val_hall = [], []
    test_ok, test_hall = [], []
    for s in all_samples:
        if s['label'] == 'ok':
            if len(train_ok) < ok_train_size:
                train_ok.append(s)
            elif len(val_ok) < ok_val_size:
                val_ok.append(s)
            else:
                test_ok.append(s)
        else:
            if len(train_hall) < hall_train_size:
                train_hall.append(s)
            elif len(val_hall) < hall_val_size:
                val_hall.append(s)
            else:
                test_hall.append(s)


    X_train = train_ok + train_hall
    random.seed(42)
    random.shuffle(X_train)
    X_val = val_ok + val_hall
    random.seed(42)
    random.shuffle(X_val)
    X_test = test_ok + test_hall
    random.seed(42)
    random.shuffle(X_test)
    print("len(X_train)", len(X_train), "len(X_val)", len(X_val), "len(X_test)", len(X_test))
    return X_train, X_val, X_test


