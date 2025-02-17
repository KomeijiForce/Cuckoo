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

def test_squad(path):
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    tagger = AutoModelForTokenClassification.from_pretrained(path).to(device)

    bar = tqdm(load_dataset("rajpurkar/squad")["validation"])

    accs, f1s = [], []

    for data in bar:
        context = data["context"]
        question = data["question"]
        answers = data["answers"]

        text = f"User:\n\n{context}\n\nQuestion: {question}\n\nAssistant:\n\nAnswer:"
        text = " ".join(word_tokenize(text))

        if len(answers["text"]) > 0:

            if len(tokenizer.tokenize(text)) + 2 <= 512:

                inputs = tokenizer(text, return_tensors="pt").to(device)
                probs = tagger(**inputs).logits[0].softmax(-1)

                tag_predictions = probs.argmax(-1)

                predictions = [tokenizer.decode(inputs.input_ids[0, seq[0]:seq[1]]).strip() for seq in find_sequences(tag_predictions)]

                if len(predictions) > 0:
                    pred = Counter(predictions).most_common(1)[0][0]
                else:
                    idx = 1+probs[1:-1, 0].argmax(0).item()
                    ids = [inputs.input_ids[0, idx]]
                    for jdx in range(idx+1, inputs.input_ids.shape[1]):
                        if not tokenizer.decode(inputs.input_ids[0, jdx]).startswith(" "):
                            ids.append(inputs.input_ids[0, jdx])
                        else:
                            break
                    pred = tokenizer.decode(ids).strip()
                    
                accs.append(pred in answers["text"])
                f1s.append(np.max([f1_score(pred, answer) for answer in answers["text"]]))

                bar.set_description(f"Accuracy = {np.mean(accs)*100:.4}% F1 = {np.mean(f1s)*100:.4}%")
                
    return np.mean(f1s) * 100

device = torch.device("cuda:0")
test_squad("models/cuckoo-squad.32shot")