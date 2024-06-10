import transformers
import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerConfig


class BERT(nn.Module):
    def __init__(self, num_classes=1, dropout_prob=0.5):
        super(BERT, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained("bert-base-uncased")

        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert_model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

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


class RoBERTa(nn.Module):
    def __init__(self, num_classes=1, dropout_prob=0.5):
        super(RoBERTa, self).__init__()
        self.roberta_model = transformers.RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids=None):
        outputs = self.roberta_model(ids, attention_mask=mask)
        hidden_state = outputs.last_hidden_state  # last hidden state
        cls_token_state = hidden_state[:, 0]  # [CLS] token

        cls_token_state = self.dropout(cls_token_state)
        out = self.out(cls_token_state)
        return out
