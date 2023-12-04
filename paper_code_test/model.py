import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps" if torch.cuda.is_available()  else "cpu")


class lstm_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=6, num_layers=1, batch_first=True)
        self.linear = nn.Linear(6, 6)
        self.relu = torch.nn.ReLU()

        self.hidden_dim = 6
        self.n_layers = 1
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()

        x, _ = self.lstm(x, (h0.detach(), c0.detach()))
        x = self.relu(x)
        x = self.linear(x)
        return x[:, -1, :]

class mlp_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.softmax(x)
        x = self.sigmoid(x)
        return x

# t_net = lstm_model()
# c_net = mlp_model()
# test_data = torch.randn(10, 6)
# print(c_net(test_data))