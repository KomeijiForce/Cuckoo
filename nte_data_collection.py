from datasets import load_dataset
import random, jsonlines
import numpy as np
from transformers import AutoTokenizer
from tqdm.notebook import tqdm
import spacy
from nltk.corpus import stopwords
import string  

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
corpora = load_dataset("teven/c4_15M")

dataset = []
start, end = 0, 100

for idx, data in enumerate(corpora['train']):
    
    if idx < start:
        continue
    
    if idx == end:
        break
        
    dataset.append({"text": data["text"]})
    
stops = stopwords.words('english') + list(string.punctuation)

bar = tqdm(dataset)

data_count = 0
empty_count = 0

with jsonlines.open("cuckoo.c4.example.json", "w") as writer:
    for data in bar:
        if len(tokenizer.tokenize(data["text"])) <= 500:
            res = nlp(data["text"])
            words_text = [word.text for word in res]
            for chunk in res.noun_chunks:
                if chunk.text.lower() not in stops:
                    chunk_seq = words_text[chunk.start:chunk.end]
                    words = words_text[:chunk.start]
                    labels = ["O" for _ in range(chunk.start)]
                    flag = False
                    for idx in range(len(words_text)-len(chunk_seq)):
                        if words[idx:idx+len(chunk_seq)] == chunk_seq:
                            flag = True
                            labels[idx] = "B"
                            for jdx in range(idx+1, idx+len(chunk_seq)):
                                labels[jdx] = "I"
                    if flag:
                        data_ie = {"words": words, "ner": labels}
                        writer.write(data_ie)
                        data_count += 1
                    elif np.random.rand() < 0.01:
                        data_ie = {"words": words, "ner": labels}
                        writer.write(data_ie)
                        data_count += 1
                        empty_count += 1
                    bar.set_description(f"#Data={data_count} #Empty={empty_count}")
