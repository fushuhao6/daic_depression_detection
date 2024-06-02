import transformers
import torch.nn as nn
from transformers import LongformerModel, LongformerConfig


class BERT(nn.Module):
    def __init__(self, num_classes=1, version='bert', dropout_prob=0.5):
        super(BERT, self).__init__()
        self.version = version
        if version == 'bert':
            self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert_model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids):
        if self.version == 'bert':
            _, o2 = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        else:
            outputs = self.bert_model(ids, attention_mask=mask, return_dict=False)
            hidden_state = outputs[0]  # last hidden state
            o2 = hidden_state[:, 0]  # [CLS] token

        o2 = self.dropout(o2)
        out = self.out(o2)
        return out


class LongformerClassifier(nn.Module):
    def __init__(self, num_classes=1, dropout_prob=0.5):
        super(LongformerClassifier, self).__init__()
        self.longformer_model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids=None):
        attention_mask = mask.clone()
        attention_mask[:, 0] = 2  # CLS token attention score
        outputs = self.longformer_model(ids, attention_mask=attention_mask)
        o2 = outputs[0][:, 0, :]  # Take the output corresponding to the [CLS] token
        o2 = self.dropout(o2)
        out = self.out(o2)
        return out
