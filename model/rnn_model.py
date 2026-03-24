import torch
import torch.nn as nn


class RNNPolicy(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 2,
        dropout: float = 0.1,
        nonlinearity: str = "tanh",
        bidirectional: bool = False,
    ):
        super().__init__()

        rnn_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            nonlinearity=nonlinearity,
            bidirectional=bidirectional,
        )

        direction_factor = 2 if bidirectional else 1

        self.head = nn.Sequential(
            nn.Linear(hidden_size * direction_factor, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        returns: (batch, seq_len, output_size)
        """
        rnn_out, _ = self.rnn(x)          # (B, T, H)
        out = self.head(rnn_out)          # (B, T, output_size)
        return out