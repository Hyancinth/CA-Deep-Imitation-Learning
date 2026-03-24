import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==========================================
# 3. Model Definition
# ==========================================
class CollisionAvoidanceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq_len, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size  # store this
        self.seq_len = seq_len        # store this
        
        # dropout only applies between LSTM layers, so only useful if num_layers > 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)  # apply after LSTM output
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # If data comes in flattened (batch, seq_len*input_size) or (batch, seq_len)
        # reshape it back to (batch, seq_len, input_size)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        return self.fc2(out)