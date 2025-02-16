# Cuckoo 🐦

Cuckoo is a small (300M) information extraction (IE) model that imitates the next token prediction paradigm of large language models. Instead of retrieving from the vocabulary, Cuckoo predicts the next tokens by tagging them in the given input context as shown below:

![cuckoo](https://github.com/user-attachments/assets/d000f275-82a7-4939-aca8-341c61a774dc)

Cuckoo is substantially different from previous IE pre-training because it can use any text resource to enhance itself, especially by taking a free ride on data curated for LLMs!

![image](https://github.com/user-attachments/assets/f4106f82-6c07-4961-a654-eca7d69428a6)


Currently, we open-source checkpoints of Cuckoos that are pre-trained on:

1) 100M next tokens extraction (NTE) instances converted from C4. ([Cuckoo-C4](https://huggingface.co/KomeijiForce/Cuckoo-C4) 🐦)

2) Cuckoo-C4 + 2.6M next token extraction (NTE) instances converted from a supervised fine-tuning dataset, TuluV3. ([Cuckoo-C4-Instruct](https://huggingface.co/KomeijiForce/Cuckoo-C4-Instruct) 🐦🛠️)

3) Cuckoo-C4-Instruct + MultiNERD, MetaIE, NuNER, MRQA (excluding SQuAD, DROP). ([Cuckoo-C4-Rainbow](https://huggingface.co/KomeijiForce/Cuckoo-C4-Rainbow) 🌈🐦🛠️)

4) Cuckoo-C4-Rainbow + Multiple NER Datasets, WizardLM Dataset, Multiple Choice QA Datasets, MMLU, SQuAD, DROP, MNLI, SNLI. ([Cuckoo-C4-Super-Rainbow](https://huggingface.co/KomeijiForce/Cuckoo-C4-Super-Rainbow) 🦸🌈🐦🛠️)

## Performance Demonstration 🚀

Begin your journey with Cuckoo to experience unimaginable adaption efficiency to all kinds of IE tasks!

|                      | CoNLL2003 | BioNLP2004 | MIT-Restaurant | MIT-Movie | Avg. | CoNLL2004 | ADE | Avg. | SQuAD | SQuAD-V2 | DROP | Avg. |
|----------------------|-----------|-----------|----------------|-----------|------|-----------|-----|------|-------|----------|------|------|
| OPT-C4-TuluV3        | 50.24     | 39.76     | 58.91          | 56.33     | 50.56 | 47.14     | 45.66 | 46.40 | 39.80 | 53.81    | 31.00 | 41.54 |
| RoBERTa              | 33.75     | 32.91     | 62.15          | 58.32     | 46.80 | 34.16     | 2.15  | 18.15 | 31.86 | 48.55    | 9.16  | 29.86 |
| MRQA                 | 72.45     | 55.93     | 68.68          | 66.26     | 65.83 | 66.23     | 67.44 | 66.84 | 80.07 | 66.22    | 54.46 | 66.92 |
| MultiNERD            | 66.78     | 54.62     | 64.16          | 66.30     | 60.59 | 57.52     | 45.10 | 51.31 | 42.85 | 50.99    | 30.12 | 41.32 |
| NuNER                | 74.15     | 56.36     | 68.57          | 64.88     | 65.99 | 65.12     | 63.71 | 64.42 | 61.60 | 52.67    | 37.37 | 50.55 |
| MetaIE               | 71.33     | 55.63     | 70.08          | 65.23     | 65.57 | 64.81     | 64.40 | 64.61 | 74.59 | 62.54    | 30.73 | 55.95 |
| Cuckoo 🐦🛠️            | 73.60     | 57.00     | 67.63          | 67.12     | 66.34 | 69.57     | 71.70 | 70.63 | 77.47 | 64.06    | 54.25 | 65.26 |
| └─ Only Pre-train 🐦    | 72.46     | 55.87     | 66.87          | 67.23     | 65.61 | 68.14     | 69.39 | 68.77 | 75.64 | 63.36    | 52.81 | 63.94 |
| └─ Only Post-train   | 72.80     | 56.10     | 66.02          | 67.10     | 65.51 | 68.66     | 69.75 | 69.21 | 77.05 | 62.39    | 54.80 | 64.75 |
| Rainbow Cuckoo 🌈🐦🛠️  | 79.94     | 58.39     | 70.30          | 67.00     | **68.91** | 70.47     | 76.05 | **73.26** | 86.57 | 69.41    | 64.64 | **73.54** |

## Quick Experience with Cuckoo in Next Tokens Extraction ⚡

We recommend using the strongest Super Rainbow Cuckoo 🦸🌈🐦🛠️ for zero-shot extraction. You can directly run the cases below in ```case_next_tokens_extraction.py```.

1️⃣ First load the model and the tokenizers

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import spacy

nlp = spacy.load("en_core_web_sm")

device = torch.device("cuda:0")
path = f"KomeijiForce/Cuckoo-C4-Super-Rainbow"
tokenizer = AutoTokenizer.from_pretrained(path)
tagger = AutoModelForTokenClassification.from_pretrained(path).to(device)
```

2️⃣ Define the next tokens extraction function
```python
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
```

3️⃣ Call the function for extraction!

Case 1: Basic entity and relation understanding

```python
text = "Tom and Jack went to their trip in Paris."

for question in [
    "What are the people mentioned here?",
    "What is the city mentioned here?",
    "Who goes with Tom together?",
    "What do Tom and Jack go to Paris for?",
    "Which city does George live in?",
]:
    text = f"User:\n\n{text}\n\nQuestion: {question}\n\nAssistant:"
    predictions = next_tokens_extraction(text)
    print(question, predictions)
```
You will get things like,
```
What are the people mentioned here? ['Tom', 'Jack']
What is the city mentioned here? ['Paris']
Who goes with Tom together? ['Jack']
What do Tom and Jack go to Paris for? ['trip']
Which city does George live in? []
```
where [] indicates Cuckoo thinks there to be no next tokens for extraction.

Case 2: Longer context

```python
passage = f'''Ludwig van Beethoven (17 December 1770 – 26 March 1827) was a German composer and pianist. He is one of the most revered figures in the history of Western music; his works rank among the most performed of the classical music repertoire and span the transition from the Classical period to the Romantic era in classical music. His early period, during which he forged his craft, is typically considered to have lasted until 1802. From 1802 to around 1812, his middle period showed an individual development from the styles of Joseph Haydn and Wolfgang Amadeus Mozart, and is sometimes characterised as heroic. During this time, Beethoven began to grow increasingly deaf. In his late period, from 1812 to 1827, he extended his innovations in musical form and expression.'''

for question in [
    "What are the people mentioned here?",
    "What is the job of Beethoven?",
    "How famous is Beethoven?",
    "When did Beethoven's middle period showed an individual development?",
]:
    text = f"User:\n\n{passage}\n\nQuestion: {question}\n\nAssistant:"
    predictions = next_tokens_extraction(text)
    print(question, predictions)
```
You will get things like,
```
What are the people mentioned here? ['Ludwig van Beethoven', 'Joseph Haydn', 'Wolfgang Amadeus Mozart']
What is the job of Beethoven? ['composer and pianist']
How famous is Beethoven? ['one of the most revered figures in the history of Western music']
When did Beethoven's middle period showed an individual development? ['1802']
```

Case 3: Knowledge quiz

```python
for obj in ["grass", "sea", "fire", "night"]:
    text = f"User:\n\nChoices:\nred\nblue\ngreen.\n\nQuestion: What is the color of the {obj}?\n\nAssistant:\n\nAnswer:"
    predictions = next_tokens_extraction(text)
    print(obj, predictions)
```
You will get things like,
```
grass ['green']
sea ['blue']
fire ['red']
night []
```
which shows Cuckoo is not extracting any plausible spans but has the knowledge to understand the context.
