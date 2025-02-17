import os
import re
import numpy as np
import pandas as pd
import requests
import json, jsonlines
import openai
from tqdm import tqdm
import subprocess
from transformers import AutoModelForTokenClassification, AutoTokenizer
from nltk import word_tokenize
import torch
from collections import defaultdict, Counter
from datasets import load_dataset

def find_sequences(lst):
    sequences = []
    i = 0
    while i < len(lst):
        if lst[i] == 0:
            start = i
            end = i
            i += 1
            while i < len(lst) and lst[i] == 1:
                end = i
                i += 1
            sequences.append((start, end+1))
        else:
            i += 1
    return sequences

def get_tag_names(dataset_name):
    url = f"https://huggingface.co/datasets/tner/{dataset_name}/raw/main/dataset/label.json"

    response = requests.get(url)

    data = json.loads(response.text)

    label_names = []

    for label_name in data:
        if label_name != "O":
            ent_name = label_name[2:].replace("-", " ").replace("_", " ")
            if ent_name != ent_name.upper() or dataset_name == "ontonotes5":
                ent_name = ent_name.capitalize()
            label_name = label_name[:2] + ent_name
        label_names.append(label_name)
        
    return label_names

def verbalizer(dataset_name, label_name):

    if dataset_name == "conll2003":
        return {
        "PER": "person",
        "LOC": "location",
        "ORG": "organization",
        "MISC": "entity other than person, location, organization",
}[label_name]
    else:
        return label_name.lower() if label_name not in ["DNA", "RNA"] else label_name

def test_ner(path, dataset_name):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tagger = AutoModelForTokenClassification.from_pretrained(path).to(device)

    tag_names = get_tag_names(dataset_name.split(".")[0])

    label_names = list(set([tag_name.split("-")[-1] for tag_name in tag_names if tag_name != "O"]))

    P, T, TP = 0, 0, 0

    bar = tqdm(load_dataset(f"tner/{dataset_name}")["test"])

    for data in bar:
        tokens, tags = data["tokens"], [tag_names[tag] for tag in data["tags"]]

        if dataset_name != "bionlp2004":
            tokens.append(".")
            tags.append("O")

        for label_name in label_names:
            golds = []
            for idx in range(len(tokens)):
                if f"B-{label_name}" == tags[idx]:
                    span = [tokens[idx]]
                    for jdx in range(idx, len(tokens)):
                        if f"I-{label_name}" == tags[jdx]:
                            span.append(tokens[jdx])
                        elif f"O" == tags[jdx]:
                            break
                    golds.append(" ".join(span))

            verbalized = verbalizer(dataset_name, label_name)

            words = word_tokenize("User:\n\n")+tokens+\
            word_tokenize("\n\nQuestion: What is the "+verbalized+" mentioned?\n\nAssistant:\n\nAnswer: The "+verbalized+" is")

            if "mit" in dataset_name:
                words.append(":")

            text = " ".join(words)

            inputs = tokenizer(text, return_tensors="pt").to(device)
            logits = tagger(**inputs).logits[0]
            probs = logits.softmax(-1)
            tag_predictions = probs.argmax(-1)

            tag_predictions = [2 if idx+1!=len(tag_predictions) and tag_prediction == 0 and tag_predictions[idx+1]==0 else tag_prediction for idx, tag_prediction in enumerate(tag_predictions)]

            preds = [tokenizer.decode(inputs.input_ids[0, seq[0]:seq[1]]).strip() for seq in find_sequences(tag_predictions)]
            preds = [pred for pred in preds if len(pred) > 0 and any([pred.split()==tokens[idx:idx+len(pred.split())] for idx in range(len(tokens)-len(pred.split()))])]

            P += len(preds)
            T += len(golds)
            TP += len([gold for gold in golds if gold in preds])
            
        prec, rec, f1 = TP/(P+1e-8)*100, TP/(T+1e-8)*100, TP*2/(T+P+1e-8)*100

        bar.set_description(f"Prec={prec:.4}, Rec={rec:.4}, F1 Score = {f1:.4}")
        
    return f1

device = torch.device("cuda:0")
test_ner("models/cuckoo-conll2003.5shot", "conll2003")
