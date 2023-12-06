import torch
import torch.nn as nn
import torch.nn.functional as F

# criterion = nn.CrossEntropyLoss()


# test_data = torch.FloatTensor([0.99])
# target = torch.FloatTensor([1])
# print(torch.log(1-test_data))
# print(criterion(test_data, target))

# loss = nn.BCELoss()
# test_data = torch.FloatTensor([0.2, 0.3])
# print(loss(test_data, torch.ones_like(test_data)))

# def g_loss(input):
#     tmp = 0
#     for i in input:
#         tmp = tmp + torch.log(1 - i)
#     print(tmp)
#     return tmp 
# test = torch.FloatTensor([0.2, 0.3])
# g_loss(test)
