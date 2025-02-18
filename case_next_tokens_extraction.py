from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import spacy

nlp = spacy.load("en_core_web_sm")

device = torch.device("cuda:0")
path = f"KomeijiForce/Cuckoo-C4-Super-Rainbow"
tokenizer = AutoTokenizer.from_pretrained(path)
tagger = AutoModelForTokenClassification.from_pretrained(path).to(device)

def next_tokens_extraction(text):

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

    text = " ".join([token.text for token in nlp(text)])

    inputs = tokenizer(text, return_tensors="pt").to(device)
    tag_predictions = tagger(**inputs).logits[0].argmax(-1)
    
    predictions = [tokenizer.decode(inputs.input_ids[0, seq[0]:seq[1]]).strip() for seq in find_sequences(tag_predictions)]
    
    return predictions

text = "Tom and Jack went to their trip in Paris."

print("Text:", text)

print('-'*100)

for question in [
    "What is the person mentioned here?",
    "What is the city mentioned here?",
    "Who goes with Tom together to Paris?",
    "What do Tom and Jack go to Paris for?",
    "Where does George live in?",
]:
    prompt = f"User:\n\n{text}\n\nQuestion: {question}\n\nAssistant:"
    predictions = next_tokens_extraction(prompt)
    print(question, predictions)
    
print('*'*100)

passage = f'''Ludwig van Beethoven (17 December 1770 – 26 March 1827) was a German composer and pianist. He is one of the most revered figures in the history of Western music; his works rank among the most performed of the classical music repertoire and span the transition from the Classical period to the Romantic era in classical music. His early period, during which he forged his craft, is typically considered to have lasted until 1802. From 1802 to around 1812, his middle period showed an individual development from the styles of Joseph Haydn and Wolfgang Amadeus Mozart, and is sometimes characterised as heroic. During this time, Beethoven began to grow increasingly deaf. In his late period, from 1812 to 1827, he extended his innovations in musical form and expression.'''

print("Passage:", passage)

print('-'*100)

for question in [
    "What is the person mentioned here?",
    "What is the job of Beethoven?",
    "How famous is Beethoven?",
    "When did Beethoven's middle period showed an individual development?",
]:
    prompt = f"User:\n\n{passage}\n\nQuestion: {question}\n\nAssistant:"
    predictions = next_tokens_extraction(prompt)
    print(question, predictions)

print('*'*100)

choices = "Choices:\nred\nblue\ngreen."

print("Choices:", choices)

print('-'*100)

for obj in ["grass", "sea", "fire", "night"]:
    prompt = f"User:\n\n{choices}\n\nQuestion: What is the color of the {obj}?\n\nAssistant:\n\nAnswer:"
    predictions = next_tokens_extraction(prompt)
    print(f"What is the color of the {obj}?", predictions)
