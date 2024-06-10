import os.path
import numpy as np
import transformers
import torch
import json
from torch.utils.data import Dataset
import nltk
from nltk.corpus import wordnet
import random

nltk.download('wordnet')

def synonym_replacement(sentence, n):
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = wordnet.synsets(random_word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # Only replace up to n words
            break

    return ' '.join(new_words)


class BertDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512, text_tag='Synopsis', augmentation=False):
        super(BertDataset, self).__init__()
        self.data_file = data_file
        with open(data_file, 'r') as f:
            data = json.load(f)

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.target = ['PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired', 'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving']
        self.text_tag = text_tag
        self.augmentation = augmentation
        self.max_replace_word = int(0.2 * max_length)

        transcript_length = []
        all_scores = []
        for d in self.data:
            transcript_length.append(len(self.get_text(d).split()))
            score = d['PHQ8_Score'] if 'PHQ8_Score' in d else d['PHQ_Score']
            all_scores.append(score)

        self.all_scores = all_scores
        print(f"Reading {len(self.data)} data from {os.path.basename(data_file)};\n\tMax length of {text_tag} is {max(transcript_length)}, "
              f"Min length is {min(transcript_length)}, Avg length is {np.mean(transcript_length)}")

    def get_text(self, data):
        if isinstance(self.text_tag, list):
            text = [data[tag] for tag in self.text_tag]
            text = ' '.join(text)
        else:
            text = data[self.text_tag]
        return text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cur_data = self.data[index]

        text = self.get_text(cur_data)
        if self.augmentation:
            text = synonym_replacement(text, n=self.max_replace_word)

        total_score_target = cur_data['PHQ8_Score'] if 'PHQ8_Score' in cur_data else cur_data['PHQ_Score']

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs.get("token_type_ids", [0] * len(ids))
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'total_target': torch.tensor(total_score_target, dtype=torch.long),
        }


class BertClassifierDataset(BertDataset):
    def __init__(self, data_file, tokenizer, max_length=512, text_tag='Transcript', augmentation=False):
        super(BertClassifierDataset, self).__init__(data_file, tokenizer, max_length, text_tag, augmentation)

    def __getitem__(self, index):
        cur_data = self.data[index]

        text = self.get_text(cur_data)
        if self.augmentation:
            text = synonym_replacement(text, n=self.max_replace_word)

        label = cur_data['Real']

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
        }


if __name__ == '__main__':
    from torch.utils.data import DataLoader, ConcatDataset
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    train_file = '/data/DAIC/train.json'
    val_file = '/data/DAIC/val.json'
    test_file = '/data/DAIC/test.json'
    synthetic_file = '/data/synthetic_DAIC/synthetic_train.json'
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = BertDataset(train_file, tokenizer, max_length=512, text_tag=['Synopsis', 'Sentiment'])
    train_scores = train_dataset.all_scores

    val_dataset = BertDataset(val_file, tokenizer, max_length=512)
    dev_scores = val_dataset.all_scores

    test_dataset = BertDataset(test_file, tokenizer, max_length=512)
    test_scores = test_dataset.all_scores


    synthetic_dataset = BertDataset(synthetic_file, tokenizer)
    syn_scores = synthetic_dataset.all_scores

    # Create a DataFrame
    data = {
        'Score': train_scores + dev_scores + test_scores,
        'Split': ['Train'] * len(train_scores) + ['Dev'] * len(dev_scores) + ['Test'] * len(test_scores)
    }
    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Score', hue='Split', fill=True, common_norm=False, alpha=0.5)
    plt.xlabel('PHQ-8 Score', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.xlim([0, 24])
    plt.tight_layout()

    # Add custom x-tick labels
    plt.xticks([0, 5, 10, 15, 20], ['Not significant', 'Mild', 'Moderate', 'Moderately severe', 'Severe'], fontsize=16)

    # Show plot
    plt.savefig("results/data_distribution.png")


    # Create a DataFrame
    data = {
        'Score': train_scores + syn_scores + dev_scores + test_scores,
        'Split': ['Train'] * (len(train_scores) + len(syn_scores)) + ['Dev'] * len(dev_scores) + ['Test'] * len(test_scores)
    }
    df = pd.DataFrame(data)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Score', hue='Split', fill=True, common_norm=False, alpha=0.5)
    plt.xlabel('PHQ-8 Score', fontsize=18)
    plt.ylabel('Density', fontsize=18)
    plt.xlim([0, 24])
    plt.tight_layout()

    # Add custom x-tick labels
    plt.xticks([0, 5, 10, 15, 20], ['Not significant', 'Mild', 'Moderate', 'Moderately severe', 'Severe'], fontsize=16)

    # Show plot
    plt.savefig("results/data_distribution_with_synthetic.png")

