import random
import json
import torch
from tqdm import trange

def load_data(path, val_size=5000, batch_size=4, shuffle = True):
    with open(path,'r') as f:
        item_list = json.load(f)
    random.seed(0)
    random.shuffle(item_list)

    train_data = []
    for i in range(
            int((len(item_list)-val_size)/batch_size)+1
        ):
        this_batch = []
        for item in item_list[val_size:][i*batch_size : (i + 1)*batch_size]:
            this_batch.append(item)
        train_data.append(this_batch)

    valid_data = []
    for i in range(
            int(val_size/batch_size)+1
        ):
        this_batch = []
        for item in item_list[:val_size][i*batch_size : (i + 1)*batch_size]:
            this_batch.append(item)
        valid_data.append(this_batch)

    return train_data, len(item_list)-val_size, valid_data, val_size

def train(model, train_data, optimizer, batch_size, train_size):
    this_epoch_loss = 0

    model.train()
    for i in trange(len(train_data), desc = 'Batch'):
        loss = 0
        for j in range(batch_size):
            text  = train_data[i][j]['text']
            summary = train_data[i][j]['summary']
            label = train_data[i][j]['label']
            loss += model(text, summary, label)
        this_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        model.zero_grad()
    this_epoch_loss = this_epoch_loss/train_size
    print('\n training loss:{:.2f}'.format(this_epoch_loss))

def test(model, valid_data, batch_size, valid_size):
    model.eval()
    true_num = 0
    with torch.no_grad():
        for i in trange(len(valid_data), desc = 'Batch'):
            for j in range(batch_size):
                text  = valid_data[i][j]['text']
                summary = valid_data[i][j]['summary']
                label = valid_data[i][j]['label']
                if len(summary)<256:
                    pred = model.inference(text, summary)
                if pred>0.5 and label==1:
                    true_num += 1
                if pred<0.5 and label==0:
                    true_num += 1

        print(true_num, valid_size)
        valid_acc = true_num/valid_size
        print('\n dev acc:{:.4f}'.format(valid_acc))

def predict(model, valid_data, batch_size, valid_size):
    model.eval()
    pred_prob_list = []
    pred_label_list = []
    with torch.no_grad():
        for i in trange(len(valid_data), desc = 'Batch'):
            for j in range(batch_size):
                text  = valid_data[i][j]['text']
                summary = valid_data[i][j]['summary']

                pred = model.inference(text, summary)
                
                pred_prob_list.append(pred)
                if pred>0.5:
                    label = 1
                else:
                    label = 0
                pred_label_list.append(label)
    return pred_prob_list, pred_label_list