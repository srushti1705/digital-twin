"""
PyTorch LSTM Architecture for Climate Prediction (Phase 5).

Implements a configurable, stacked Long Short-Term Memory (LSTM) neural network
for 1-day ahead daily climate forecasting (rainfall & temperature).

Architecture Overview:
----------------------
1. Input Sequence: Tensor (batch_size, seq_len, input_dim)
2. Stacked LSTM:   Multi-layer recurrent feature extraction with gate operations
3. Dropout Layer:   Regularization preventing co-adaptation of hidden states
4. Fully-Connected: Linear projection (batch_size, hidden_dim) -> (batch_size, output_dim)
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class ClimateLSTMModel(nn.Module):
    """
    Configurable PyTorch Stacked LSTM Model for Climate Sequence Prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input features measured per timestep (e.g., 25 for rainfall, 21 for tmax).
        hidden_dim : int, default=64
            Number of hidden units in each LSTM layer.
        num_layers : int, default=2
            Number of stacked recurrent LSTM layers.
        output_dim : int, default=1
            Dimension of the output prediction (1 for scalar single-step regression).
        dropout : float, default=0.2
            Dropout probability applied between stacked LSTM layers and before FC head.
        bidirectional : bool, default=False
            Whether to use bidirectional LSTM (False for causal time-series forecasting).
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 1. Recurrent Stack: Stacked PyTorch LSTM
        # Requires dropout > 0 only when num_layers > 1
        lstm_dropout = dropout if num_layers > 1 else 0.0
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )

        # 2. Regularization Layer
        self.dropout = nn.Dropout(p=dropout)

        # 3. Dense Projection Head
        fc_input_dim = hidden_dim * self.num_directions
        self.fc = nn.Linear(fc_input_dim, output_dim)

        # Initialize weights for linear head
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for climate sequence prediction.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns
        -------
        torch.Tensor
            Output predictions of shape (batch_size, output_dim) or (batch_size,).
        """
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)

        # 1. Pass through Stacked LSTM
        # lstm_out shape: (batch_size, sequence_length, hidden_dim * num_directions)
        # (h_n, c_n) shape: (num_layers * num_directions, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 2. Extract final sequence timestep vector
        # Using lstm_out[:, -1, :] guarantees taking the representation at Day t-1 to predict Day t
        last_timestep_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_dim * num_directions)

        # 3. Apply Dropout Regularization
        out_dropped = self.dropout(last_timestep_out)

        # 4. Dense Linear Projection Head
        # output shape: (batch_size, output_dim)
        out = self.fc(out_dropped)

        # Squeeze single output dimension if output_dim == 1
        if self.output_dim == 1:
            out = out.squeeze(-1)  # shape: (batch_size,)

        return out


def build_rainfall_lstm_model(
    input_dim: int = 25,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2
) -> ClimateLSTMModel:
    """
    Factory function for building the default PyTorch LSTM model for Rainfall prediction.
    """
    return ClimateLSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout,
        bidirectional=False
    )


def build_temperature_lstm_model(
    input_dim: int = 21,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2
) -> ClimateLSTMModel:
    """
    Factory function for building the default PyTorch LSTM model for Maximum Temperature prediction.
    """
    return ClimateLSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout,
        bidirectional=False
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PYTORCH CLIMATE LSTM ARCHITECTURE FORWARD PASS TEST")
    print("=" * 60)

    # 1. Test Rainfall Model Configuration
    rf_model = build_rainfall_lstm_model(input_dim=25, hidden_dim=64, num_layers=2)
    print("Rainfall LSTM Model Architecture:")
    print(rf_model)
    
    batch_size = 32
    seq_len = 7
    dummy_x_rain = torch.randn(batch_size, seq_len, 25)
    out_rain = rf_model(dummy_x_rain)
    print(f"\nRainfall Forward Pass Check:")
    print(f"  - Input Shape:  {dummy_x_rain.shape}")
    print(f"  - Output Shape: {out_rain.shape}  (Expected: ({batch_size},))")

    # 2. Test Temperature Model Configuration
    temp_model = build_temperature_lstm_model(input_dim=21, hidden_dim=128, num_layers=2)
    print("\nTemperature LSTM Model Architecture:")
    print(temp_model)

    dummy_x_temp = torch.randn(batch_size, seq_len, 21)
    out_temp = temp_model(dummy_x_temp)
    print(f"\nTemperature Forward Pass Check:")
    print(f"  - Input Shape:  {dummy_x_temp.shape}")
    print(f"  - Output Shape: {out_temp.shape}  (Expected: ({batch_size},))")

    # 3. Test Autograd Backpropagation
    loss = out_temp.sum()
    loss.backward()
    print("\nAutograd Gradient Backward Pass Check: PASSED")
    print("=" * 60)
