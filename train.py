from data_util import TrainLoader, TestLoader
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
from models import Transfomer
from tqdm import tqdm

from test import testing


batch_size = 8
N = 2

model = Transfomer(input_dim=14)

train_data = TrainLoader("datasets/train_FD001.txt")
x_data = DataLoader(train_data)

optim = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)


model.train()


for epoch in range(100):
    progressbar = tqdm(x_data,desc="[Train] loss:0, step:0")
    epoch_loss = 0
    for x in progressbar:
        y = x[0, -1:]

        seq_len, bs, input_dim = x.size()
        total_loss = 0

        optim.zero_grad()
        for t in range(1,bs-1):
            x_ = x[:,t - 1:t + 2, 2:-1]
            y = x[:,t, -1:] 

            x_train_tensors = Variable(x_)
            y_train_tensors = Variable(y)
            out = model(x_train_tensors,t)

            loss = criterion(out, y_train_tensors)
            total_loss += loss.item()

            loss = loss / (bs-2)
            loss.backward()
            if t == x.size(1) - 2:
                optim.step()
                optim.zero_grad()
            
        epoch_loss += total_loss / bs
        progressbar.set_description(f"[Train] loss:{epoch_loss/bs}, epoch:{epoch}")
    with torch.no_grad():
        rmse = testing(model)
    print("Epoch: %d, training loss: %1.5f, testing rmse: %1.5f" % (epoch, epoch_loss / 100, rmse))

