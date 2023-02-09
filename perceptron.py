import torch
from torch import nn
from neural_network_linear import NeuralNetworkLinear
from neural_network_tanh import NeuralNetworkTanh

class Perceptron:
    def __init__(self, train_dataloader=None, test_dataloader = None, batch_size = 0, type = None, loss_fn=nn.MSELoss(), path=None):
        if path is None:
            self.train_dataloader = train_dataloader
            self.test_dataloader = test_dataloader

            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

            if type == "Tanh":
                self.model = NeuralNetworkTanh().to(self.device)
            else:
                self.model = NeuralNetworkLinear().to(self.device)

            self.loss_fn = loss_fn
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            if type == "Tanh":
                self.model = NeuralNetworkTanh().to(self.device)
            else:
                self.model = NeuralNetworkLinear().to(self.device)
            self.load(path)


    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, dataloader):
        ret = []
        self.model.eval()
        with torch.no_grad():
            for X in dataloader.dataset.tensors:
                X = X.to(self.device)
                pred = self.model(X)
                pred_arr = pred.cpu().detach().numpy()
                ret.append(pred_arr)
        return pred_arr

    def train(self, epochs):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self._train(self.train_dataloader, self.model, self.loss_fn, self.optimizer)
            if self.test_dataloader is not None:
                self._test(self.test_dataloader, self.model, self.loss_fn)
        print("Done!")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Saved PyTorch Model State to {path}")

    def _train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def _test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                y_arr = y.cpu().detach().numpy()
                pred_arr = pred.cpu().detach().numpy()

                for i in range(0, size):
                    print(f"Expected : {y_arr[i]}, and predicted : {pred_arr[i]}\n")
        test_loss /= num_batches
        print(f"Test Error: Avg loss: {test_loss:>8f} \n")