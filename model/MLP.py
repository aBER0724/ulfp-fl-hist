import torch
from torch import nn
import torch.nn.functional as F

class PrivateFraudNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.GroupNorm(1, hidden_dim)
        )
        # make the number of hidden dim layers configurable
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.GroupNorm(1, hidden_dim))
            self.layers.append(nn.Dropout(0.5))

        # final layer
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out = self.input(x)
        for layer in self.layers:
            out = layer(out)
        return self.fc(out)

# Simple MLP for tabular data (keep existing if needed elsewhere, or remove if PrivateFraudNet replaces it)
class MLPTabular(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPTabular, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x 