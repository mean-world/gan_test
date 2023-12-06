
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
import argparse

import model as model_class

device = torch.device("mps" if torch.cuda.is_available()  else "cpu")


if __name__ == "__main__":
    steps = 1
    epochs = 50
    data_set = "temoral"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="model epochs number", default=10)
    parser.add_argument('-m', help="discriminator, model choice mlp or cnn", default="cnn")
    parser.add_argument('-t', help="time window size", default=6)
    parser.add_argument('-d', help='path data set',default='data_set.csv')
    parser.add_argument('-l1', help="λ1", default=1.)
    parser.add_argument('-l2', help="λ2", default=1.)
    parser.add_argument('-b', help="batch size ", default=2)
    parser.add_argument('-lr', help="learning rate", default=0.0002)
    parser.add_argument('-be', help="beta1", default=0.5)
    parser.add_argument('-s', help="split dataset rate", default=0.8)



    args = parser.parse_args()


    data = pd.read_csv(args.d, index_col=None, header=0)


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

    train_set, train_label, test_set, test_label = split_data(data, args.t, args.s)


    class data_set(Dataset):
        def __init__(self, data, transform=None, target_transform=None):
            self.transform = transform
            self.target_transform = target_transform
            self.train, self.label = data

        def __len__(self):
            return self.train.size(0)

        def __getitem__(self, index):
            return self.train[index, :, :], self.label[index, :]
        
    batch_size = args.b
    train_dataset = data_set((train_set, train_label))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)



    netG = model_class.lstm_model().to(device)
    # print(netG)

    if args.m =="mlp":
        netD = model_class.mlp_model().to(device)
    else:
        netD = model_class.cnn_model(args.t).to(device)
    # print(netD)

    num_epochs = args.e
    # Learning rate for optimizers
    lr = args.lr
    # Beta1 hyperparameter for Adam optimizers
    beta1 = args.be
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    criterion = nn.BCELoss()
    mse_loss = nn.MSELoss()

    def g_loss_fn(input):
        tmp = 0
        for i in input:
            tmp = tmp + torch.log(1 - i)
        # print(tmp)
        return tmp 
    num_epochs = args.e
    λ1 = args.l1
    λ2 = args.l2
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):

            netD.zero_grad()
            #create fake time series
            fake = netG(data)
            
            #concat with real data
            fake_data = torch.cat((data, fake.view(fake.size(0), 1, fake.size(1))), dim=1)
            real_data = torch.cat((data, target.view(target.size(0), 1, target.size(1))), dim=1)

            #D judgment
            #real
            real_data_output = netD(real_data[0, :, :])
            
            #batch
            for i in range(1, real_data.size(0)):
                real_data_output = torch.cat((real_data_output, netD(real_data[i, :, :])), dim=0)
            #fake
            fake_data_output = netD(fake_data[0, :, :].detach())
            #batch
            for i in range(1, fake_data.size(0)):
                fake_data_output = torch.cat((fake_data_output, netD(fake_data[i, :, :].detach())), dim=0)

            real_loss = criterion(real_data_output, torch.ones_like(real_data_output))
            fake_loss = criterion(fake_data_output, torch.zeros_like(fake_data_output))
            D_loss = real_loss + fake_loss
            D_loss.backward()
            optimizerD.step()
            
            
            netG.zero_grad()
            #g loss
            g_mse = mse_loss(fake, target)
            fake_data_output = netD(fake_data[0, :, :])
            for i in range(1, fake_data.size(0)):
                torch.cat((fake_data_output, netD(fake_data[i, :, :])), dim=0)
            g_loss = g_loss_fn(fake_data_output)
            
            G_loss = g_mse + g_loss
            # G_loss = λ1 * g_mse + λ2 * g_loss
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