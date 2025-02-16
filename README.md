# Cuckoo :bird:

Cuckoo is a small (300M) information extraction (IE) model that imitates the next token prediction paradigm of large language models. Instead of retrieving from the vocabulary, Cuckoo predicts the next tokens by tagging them in the given input context as shown below:

![cuckoo](https://github.com/user-attachments/assets/d000f275-82a7-4939-aca8-341c61a774dc)

Currently, we open-source checkpoints of Cuckoos that are pre-trained on:

1) 100M next tokens extraction (NTE) instances converted from C4. (Cuckoo-C4)

2) Cuckoo-C4 + 2.6M next token extraction (NTE) instances converted from a supervised fine-tuning dataset, TuluV3. (Cuckoo-C4-Instruct)

3) Cuckoo-C4-Instruct + MultiNERD, MetaIE, NuNER, MRQA (excluding SQuAD, DROP). (Cuckoo-C4-Rainbow)

4) Cuckoo-C4-Rainbow + Multiple NER Datasets, WizardLM Dataset, Multiple Choice QA Datasets, MMLU, SQuAD, DROP, MNLI, SNLI. (Cuckoo-C4-Super-Rainbow)
