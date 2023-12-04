
import torch
import torch.nn.functional as F
import torch.distributed as dist

import yfinance as yf
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
from sklearn import preprocessing
import torch.optim as optim

import model as model_class

device = torch.device("mps" if torch.cuda.is_available()  else "cpu")


# Get stock DataSet.
#web crawler
#'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
# start_date = '2009-01-01'
# end_date = '2023-11-08'
# ticker = 'GOOGL'
# data = yf.download(ticker, start_date, end_date)
# # print(data)
# data.to_csv("data_set.csv", index=False)
data = pd.read_csv("data_set.csv", index_col=None, header=0)


#normalize data
def normalize(data):
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)
    return data
data = normalize(data)

#data preprocess
def split_data(stock, window_size, rate):
    data_raw = stock.to_numpy()
    data = []

    for i in range(len(data_raw) - window_size):
        data.append(data_raw[i: i + window_size])

    data = np.array(data)
    test_set_size = int(np.floor(rate * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size,:-1]
    y_train = data[:train_set_size,-1]

    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1]

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    return [x_train, y_train, x_test, y_test]

train_set, train_label, test_set, test_label = split_data(data, 6, 0.8)


class data_set(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train, self.label = data

    def __len__(self):
        return self.train.size(0)

    def __getitem__(self, index):
        return self.train[index, :, :], self.label[index, :]
    
batch_size = 1
train_dataset = data_set((train_set, train_label))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)



netG = model_class.lstm_model().to(device)
# print(netG)

netD = model_class.mlp_model().to(device)
# print(netD)

num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

criterion = nn.BCELoss()
loss_fn_G = nn.MSELoss()

mse_loss = nn.MSELoss()

def g_loss_fn(input):
    tmp = 0
    for i in input:
        tmp = tmp + torch.log(1 - i)
    # print(tmp)
    return tmp 
num_epochs = 2
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        netD.zero_grad()
        #create fake time series
        fake = netG(data)
        #concat with real data
        fake_data = torch.cat((data.view(data.size(1), data.size(2)), fake), dim=0)


        #D judgment
        #real
        real_data_output = netD(target)
        #fake
        fake_data_output = netD(fake.detach())

        real_loss = criterion(real_data_output, torch.ones_like(real_data_output))
        fake_loss = criterion(fake_data_output, torch.zeros_like(fake_data_output))
        D_loss = real_loss + fake_loss
        D_loss.backward()
        optimizerD.step()

        netG.zero_grad()
        #g loss
        g_mse = mse_loss(fake, target)
        g_loss = g_loss_fn(netD(fake))
        G_loss = g_mse + g_loss
        G_loss.backward()
        optimizerG.step()


        if batch_idx % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tg_MSE: %.4f\tg_loss: %.4f '
                    % (epoch+1, num_epochs, batch_idx, len(train_loader),
                        D_loss.item(), G_loss.item(), g_mse, g_loss))


    #test
    with torch.no_grad():
        output = netG(test_set)
        loss = mse_loss(output, test_label)
        print(f"[{epoch+1}] Validation loss {loss:.4f}")
        

