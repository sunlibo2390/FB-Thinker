from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

class RM_model(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.bert = BertModel.from_pretrained('bert-base-chinese').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.linear = nn.Linear(768, 1).to(self.device)
        self.softmax = nn.Softmax(dim = 0).to(self.device)
        self.loss_fn = nn.BCELoss()

    def forward(self, text, summary, label):
        encoded_input = self.tokenizer(text[:500-len(summary)] + '[SEP]' + summary, return_tensors = 'pt', truncation=False).to(self.device)
        output = self.bert(**encoded_input)
        linear_output = self.linear(output.pooler_output)
        pred = torch.sigmoid(linear_output)

        loss = self.loss_fn(
            pred,
            torch.tensor([[float(label)]]).to(self.device)
        )
        return loss
    
    def inference(self, text, summary):#此处从候选句中选择一个结果，hyps为字符串列表
        encoded_input = self.tokenizer(text[:500-len(summary)] + '[SEP]' + summary, return_tensors = 'pt', truncation=False).to(self.device)
        output = self.bert(**encoded_input)
        linear_output = self.linear(output.pooler_output)
        pred = torch.sigmoid(linear_output)
        return pred
    

class RM_model_bi(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.bert = BertModel.from_pretrained('bert-base-chinese').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.linear = nn.Linear(768, 2).to(self.device)
        self.softmax = nn.Softmax(dim = 0).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, text, summary, label):
        encoded_input = self.tokenizer(text[:500-len(summary)] + '[SEP]' + summary, return_tensors = 'pt', truncation=False).to(self.device)
        output = self.bert(**encoded_input)
        linear_output = self.linear(output.pooler_output)
        sfmax = self.softmax(linear_output)

        loss = self.loss_fn(
            sfmax,
            torch.tensor([label]).to(self.device)
        )
        return loss
    
    def inference(self, text, summary):#此处从候选句中选择一个结果，hyps为字符串列表
        encoded_input = self.tokenizer(text[:500-len(summary)] + '[SEP]' + summary, return_tensors = 'pt', truncation=False).to(self.device)
        output = self.bert(**encoded_input)
        linear_output = self.linear(output.pooler_output)
        pred_idx = torch.argmax(linear_output)
        return pred_idx

