import torch
import torch.nn as nn
from data.model_data import X_COLUMNS, Y_COLUMNS

class basicAnn(nn.Module):
    def __init__(self):
        super().__init__()

        ### input layer
        self.input = nn.Linear(len(X_COLUMNS), 64)

        ### hidden layers
        self.hidden1 = nn.Linear(64, 128)
        self.hidden2 = nn.Linear(128, 64)

        ### output layer
        self.output = nn.Linear(64, len(Y_COLUMNS))
    
    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        
        return self.output(x)
    