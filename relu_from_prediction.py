from perceptron import Perceptron
import torch
from torch.utils.data import TensorDataset, DataLoader

perceptron_2x_plus_1 = Perceptron(path="2x_plus_1.pth", type="Linear")

perceptron_0x_plus_0 = Perceptron(path="0x_plus_0.pth", type="Linear")

perceptron_pos_neg = Perceptron(path="pos_neg.pth", type="Tanh")

inputs = [[-4], [1], [6]]

batch_size = 3

tensor_inputs = torch.Tensor(inputs)

dataset = TensorDataset(tensor_inputs) # create your datset
dataloader = DataLoader(dataset, batch_size=batch_size) # create your dataloader

results = perceptron_pos_neg.predict(dataloader)
print(results)

inputs_2x_plus_1 = []
inputs_0x_plus_0 = []

for i in range(0,len(results)):
    if results[i][0] < 0:
        inputs_0x_plus_0.append(inputs[i])
    else:
        inputs_2x_plus_1.append(inputs[i])

tensor_inputs_0x_plus_0 = torch.Tensor(inputs_0x_plus_0)

dataset_0x_plus_0 = TensorDataset(tensor_inputs_0x_plus_0) # create your datset
dataloader_0x_plus_0 = DataLoader(dataset_0x_plus_0) # create your dataloader

results_0x_plus_0 = perceptron_0x_plus_0.predict(dataloader_0x_plus_0)
print(results_0x_plus_0)

tensor_inputs_2x_plus_1 = torch.Tensor(inputs_2x_plus_1)

dataset_2x_plus_1 = TensorDataset(tensor_inputs_2x_plus_1) # create your datset
dataloader_2x_plus_1 = DataLoader(dataset_2x_plus_1) # create your dataloader

results_2x_plus_1 = perceptron_2x_plus_1.predict(dataloader_2x_plus_1)
print(results_2x_plus_1)

for i in range(0, len(tensor_inputs_0x_plus_0)):
    print(f"Prediction for {tensor_inputs_0x_plus_0[i][0]}: {results_0x_plus_0[i][0]}")

for i in range(0, len(tensor_inputs_2x_plus_1)):
    print(f"Prediction for {tensor_inputs_2x_plus_1[i][0]}: {results_2x_plus_1[i][0]}")