from perceptron import Perceptron
import torch
from torch.utils.data import TensorDataset, DataLoader

perceptron_2x_plus_1 = Perceptron(path="2x_plus_1.pth", type="Linear")

perceptron_0x_plus_0 = Perceptron(path="0x_plus_0.pth", type="Linear")

perceptron_0x_plus_5 = Perceptron(path="0x_plus_5.pth", type="Linear")

perceptron_pos_neg = Perceptron(path="pos_neg.pth", type="Tanh")

perceptron_under_above_5 = Perceptron(path="under_above_5.pth", type="Tanh")

inputs = [[-4], [1], [6]]

batch_size = 3

tensor_inputs = torch.Tensor(inputs)

dataset = TensorDataset(tensor_inputs) # create your datset
dataloader = DataLoader(dataset, batch_size=batch_size) # create your dataloader

results_0x_plus_0 = perceptron_0x_plus_0.predict(dataloader)
print(results_0x_plus_0)

results_2x_plus_1 = perceptron_2x_plus_1.predict(dataloader)
print(results_2x_plus_1)

results_0x_plus_5 = perceptron_0x_plus_5.predict(dataloader)
print(results_0x_plus_5)

pos_neg_tensor_inputs = torch.Tensor(results_2x_plus_1)
pos_neg_dataset = TensorDataset(pos_neg_tensor_inputs) # create your datset
pos_neg_dataloader = DataLoader(pos_neg_dataset, batch_size=batch_size) # create your dataloader

pos_neg_results = perceptron_pos_neg.predict(pos_neg_dataloader)
print(pos_neg_results)
under_above_5_results = perceptron_under_above_5.predict(pos_neg_dataloader)
print(under_above_5_results)

for i in range(0, len(inputs)):
    if (pos_neg_results[i][0] < 0):
        print(f"Prediction for {inputs[i][0]}: {results_0x_plus_0[i][0]}")
    else:
        if (under_above_5_results[i][0] < 0):
            print(f"Prediction for {inputs[i][0]}: {results_2x_plus_1[i][0]}")
        else:
            print(f"Prediction for {inputs[i][0]}: {results_0x_plus_5[i][0]}")
