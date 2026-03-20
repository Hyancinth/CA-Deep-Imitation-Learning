import torch
import torch.nn as nn

class basicAnn(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        ### input layer
        self.input = nn.Linear(input_size, 64)

        ### hidden layers
        self.hidden1 = nn.Linear(64, 128)
        self.hidden2 = nn.Linear(128, 64)

        ### output layer
        self.output = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        
        return self.output(x)
    