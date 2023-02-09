import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

train_inputs = [[-5], [0], [5]]
train_outputs = [[-9], [1], [11]]

tensor_train_inputs = torch.Tensor(train_inputs)
tensor_train_outputs = torch.Tensor(train_outputs)

test_inputs = [[-4], [1], [6]]
test_outputs = [[-7], [3], [13]]

batch_size = 3

tensor_test_inputs = torch.Tensor(test_inputs)
tensor_test_outputs = torch.Tensor(test_outputs)

train_dataset = TensorDataset(tensor_train_inputs,tensor_train_outputs) # create your datset
train_dataloader = DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

test_dataset = TensorDataset(tensor_test_inputs,tensor_test_outputs) # create your datset
test_dataloader = DataLoader(test_dataset, batch_size=batch_size) # create your dataloader

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            y_arr = y.cpu().detach().numpy()
            pred_arr = pred.cpu().detach().numpy()

            for i in range(0, size):
                print(f"Expected : {y_arr[i]}, and predicted : {pred_arr[i]}\n")
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")

epochs = 2000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")