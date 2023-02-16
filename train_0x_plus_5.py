from perceptron import Perceptron
import torch
from torch.utils.data import TensorDataset, DataLoader

train_inputs = [[-5], [0], [5]]
train_outputs = [[5], [5], [5]]

tensor_train_inputs = torch.Tensor(train_inputs)
tensor_train_outputs = torch.Tensor(train_outputs)

test_inputs = [[-4], [1], [6]]
test_outputs = [[5], [5], [5]]

batch_size = 3

tensor_test_inputs = torch.Tensor(test_inputs)
tensor_test_outputs = torch.Tensor(test_outputs)

train_dataset = TensorDataset(tensor_train_inputs,tensor_train_outputs) # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

test_dataset = TensorDataset(tensor_test_inputs,tensor_test_outputs) # create your datset
test_dataloader = DataLoader(test_dataset, batch_size=batch_size) # create your dataloader

p = Perceptron(train_dataloader, test_dataloader, batch_size = batch_size)
p.train(10000)
p.save("0x_plus_5.pth")