import time
import torch
from bert_model import RM_model_single, RM_model_bi
import numpy as np
from torch.optim import Adam, sgd
from tqdm import trange
from utils import load_data, train, test

reward = "relate" # relate, factual, compre
data_path = f'./reward_model/data/{reward}_error.json'
save_model_dir = f"./reward_model/models/{reward}"
device = torch.device('cuda')
epochs = 20
batch_size = 32
val_size = 5000
lr = 1e-5

train_data, train_size, valid_data, valid_size = load_data(path=data_path, batch_size=batch_size,val_size=val_size)
 
model = RM_model_bi(device)
optimizer = Adam(model.parameters(), lr = lr)


print('is training model...')
test(model, valid_data, batch_size, valid_size)
for epoch in trange(epochs, desc = 'Epoch'):
    begin = time.time()
    
    train(model, train_data, optimizer, batch_size, train_size)
    test(model, valid_data, batch_size, valid_size)

    end = time.time()
    print(f'Epoch {epoch} time using:{end - begin}')
    torch.save(model, f'{save_model_dir}/bert-base-finetuned-{epoch}.pt')


        