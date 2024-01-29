import time
import torch
from bert_model import RM_model, RM_model_bi
import numpy as np
from torch.optim import Adam, sgd
from tqdm import trange
from utils import load_data, train, test
import os

data_path = './reward_model/data/factual_error.json'
save_model_dir = "./reward_model/models/factual"
device = torch.device('cuda')
epochs = 20
batch_size = 32
val_size = 1
lr = 1e-5

train_data, train_size, valid_data, valid_size = load_data(path=data_path, batch_size=batch_size,val_size=val_size)
 
model = torch.load(os.path.join(save_model_dir, "bert-base-finetuned.pt"), device)
model.device = device
optimizer = Adam(model.parameters(), lr = lr)


print('is training model...')
test(model, train_data, batch_size, train_size)
        