from torch import nn

class NeuralNetworkTanh(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.stack = nn.Sequential(
                nn.Linear(1, 1),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.stack(x)
            return logits