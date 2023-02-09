from torch import nn

class NeuralNetworkLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.stack = nn.Sequential(
                nn.Linear(1, 1)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.stack(x)
            return logits