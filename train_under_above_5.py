from perceptron import Perceptron
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math

def sig(x):
    return 1/(1 + np.exp(-x))

train_inputs = [[-5], [-1], [1], [3.5], [4.9], [4.95], [5.05], [5.1], [6.5], [7], [10], [11]]
train_outputs = []
for x in train_inputs:
    train_outputs.append([math.tanh(x[0] - 5)])

tensor_train_inputs = torch.Tensor(train_inputs)
tensor_train_outputs = torch.Tensor(train_outputs)

test_inputs = [[1], [6], [11]]
test_outputs = []
for x in test_inputs:
    test_outputs.append([math.tanh(x[0] - 5)])

batch_size = 12

tensor_test_inputs = torch.Tensor(test_inputs)
tensor_test_outputs = torch.Tensor(test_outputs)

train_dataset = TensorDataset(tensor_train_inputs,tensor_train_outputs) # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

test_dataset = TensorDataset(tensor_test_inputs,tensor_test_outputs) # create your datset
test_dataloader = DataLoader(test_dataset, batch_size=batch_size) # create your dataloader

p = Perceptron(train_dataloader, test_dataloader, batch_size=batch_size, type="Tanh")
p.train(10000)
p.save("under_above_5.pth")