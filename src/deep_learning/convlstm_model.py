"""
PyTorch ConvLSTM Architecture for Spatio-Temporal Climate Prediction (Phase 5).

Implements a configurable, multi-layer Convolutional LSTM (ConvLSTM) neural network
for 1-day ahead 2D gridded daily climate forecasting (rainfall & temperature).

Why ConvLSTM is Appropriate for Gridded Climate Prediction:
------------------------------------------------------------
1. Spatio-Temporal Coupling: Atmospheric processes (e.g., monsoon moisture transport,
   heatwave diffusion, frontal movement) depend simultaneously on spatial spatial proximity 
   and temporal momentum. Standard 1D LSTMs treat grid cells independently, discarding 
   spatial advection across neighboring pixels. Standard 2D CNNs treat time steps 
   independently, discarding temporal memory. ConvLSTM combines 2D spatial convolution 
   operators directly inside the LSTM gate transitions, learning spatial patterns 
   (e.g., pressure gradients, elevation boundaries) alongside temporal dynamics.
2. Topology Preservation: By processing inputs as 5D tensors (Batch, Time, Channels, 
   Latitude, Longitude), ConvLSTM preserves the regular 0.25° x 0.25° IMD geographical 
   grid canvas without loss of spatial structure or flattening artifacts.
"""

from typing import Tuple, List, Union, Optional
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single Convolutional LSTM (ConvLSTM) Cell.

    Replaces matrix multiplications in standard LSTM gate equations with 2D convolutions.
    Computes all 4 gates (input, forget, cell, output) simultaneously via a combined 2D convolution.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        bias: bool = True
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input channels.
        hidden_dim : int
            Number of hidden state channels.
        kernel_size : int or Tuple[int, int], default=3
            Size of the 2D convolutional kernel.
        bias : bool, default=True
            Whether to add a learnable bias to the convolution layers.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Padding chosen as kernel_size // 2 to maintain spatial height & width
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # Combined 2D convolution for input and hidden state -> 4 * hidden_dim channels (i, f, c, o)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=bias
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes convolution weights using Xavier uniform and zero bias."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for a single timestep.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch_size, input_dim, height, width).
        state : Tuple[torch.Tensor, torch.Tensor], optional
            Previous (h, c) state tensors of shape (batch_size, hidden_dim, height, width).
            If None, initialized to zeros.

        Returns
        -------
        h_next, (h_next, c_next) : Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Updated hidden state and state tuple for the next timestep.
        """
        batch_size, _, height, width = x.size()

        if state is None:
            h_prev = torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device, dtype=x.dtype)
            c_prev = torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device, dtype=x.dtype)
        else:
            h_prev, c_prev = state

        # Concatenate input tensor and previous hidden state along channel dimension
        combined = torch.cat([x, h_prev], dim=1)  # shape: (batch_size, input_dim + hidden_dim, H, W)

        # Apply 2D Convolution to compute all gate projections simultaneously
        conv_out = self.conv(combined)  # shape: (batch_size, 4 * hidden_dim, H, W)

        # Split into gate components: input, forget, cell candidate, output
        cc_i, cc_f, cc_c, cc_o = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        c_candidate = torch.tanh(cc_c)
        o = torch.sigmoid(cc_o)

        # Update cell state and hidden state
        c_next = f * c_prev + i * c_candidate
        h_next = o * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class ClimateConvLSTMModel(nn.Module):
    """
    Configurable PyTorch Multi-Layer ConvLSTM Model for Spatio-Temporal Climate Forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[int, List[int]] = 64,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bias: bool = True
    ):
        """
        Parameters
        ----------
        input_dim : int
            Number of input feature channels (e.g., 25 for rainfall, 21 for tmax).
        hidden_dims : int or List[int], default=64
            Number of hidden channels in each ConvLSTM layer.
        kernel_size : int or Tuple[int, int], default=3
            Size of 2D spatial convolution filters.
        num_layers : int, default=2
            Number of stacked ConvLSTM layers.
        output_dim : int, default=1
            Number of target output channels (1 for single-variable grid forecasting).
        dropout : float, default=0.2
            Spatial dropout probability applied between layers and before final Conv head.
        bias : bool, default=True
            Whether to include bias in spatial convolutions.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout

        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * num_layers
        else:
            if len(hidden_dims) != num_layers:
                raise ValueError(f"Length of hidden_dims ({len(hidden_dims)}) must match num_layers ({num_layers}).")
            self.hidden_dims = hidden_dims

        # 1. Build Stacked ConvLSTM Layers
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else self.hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=kernel_size,
                    bias=bias
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        # 2. Regularization Layer
        self.dropout = nn.Dropout2d(p=dropout)

        # 3. 2D Spatial Projection Head (Projects final hidden channels -> output channels)
        self.conv_out = nn.Conv2d(
            in_channels=self.hidden_dims[-1],
            out_channels=output_dim,
            kernel_size=1,  # 1x1 convolution across spatial positions
            padding=0,
            bias=bias
        )

        nn.init.xavier_uniform_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spatio-temporal climate prediction.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence tensor of shape (batch_size, sequence_length, num_channels, height, width)
            or (batch_size, sequence_length, height, width, num_channels).

        Returns
        -------
        torch.Tensor
            Predicted 2D spatial grid map of shape (batch_size, height, width) if output_dim == 1,
            or (batch_size, output_dim, height, width) if output_dim > 1.
        """
        # Automatically handle input format (Batch, Time, Height, Width, Channels) -> (Batch, Time, Channels, Height, Width)
        if x.ndim == 5 and x.size(2) > x.size(4):
            # Formatted as (B, T, H, W, C) -> transpose to (B, T, C, H, W)
            x = x.permute(0, 1, 4, 2, 3)

        batch_size, sequence_length, _, height, width = x.size()

        # Initialize hidden and cell states for all stacked ConvLSTM layers
        states = [None] * self.num_layers

        # Sequential processing across historical time steps
        for t in range(sequence_length):
            input_tensor = x[:, t, :, :, :]  # shape: (batch_size, channels, H, W)

            for layer_idx, cell in enumerate(self.cell_list):
                h_next, states[layer_idx] = cell(input_tensor, states[layer_idx])
                input_tensor = h_next

        # Extract final hidden state representation from the top ConvLSTM layer
        final_h = states[-1][0]  # shape: (batch_size, hidden_dims[-1], H, W)

        # Apply 2D Spatial Dropout
        h_dropped = self.dropout(final_h)

        # Project via 1x1 Convolution to target output grid
        out = self.conv_out(h_dropped)  # shape: (batch_size, output_dim, H, W)

        # Squeeze single output channel dimension if output_dim == 1
        if self.output_dim == 1:
            out = out.squeeze(1)  # shape: (batch_size, H, W)

        return out


def build_rainfall_convlstm_model(
    input_dim: int = 25,
    hidden_dims: Union[int, List[int]] = 64,
    num_layers: int = 2,
    kernel_size: int = 3,
    dropout: float = 0.2
) -> ClimateConvLSTMModel:
    """
    Factory function for building default PyTorch ConvLSTM model for Rainfall prediction.
    """
    return ClimateConvLSTMModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        kernel_size=kernel_size,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout
    )


def build_temperature_convlstm_model(
    input_dim: int = 21,
    hidden_dims: Union[int, List[int]] = 64,
    num_layers: int = 2,
    kernel_size: int = 3,
    dropout: float = 0.2
) -> ClimateConvLSTMModel:
    """
    Factory function for building default PyTorch ConvLSTM model for Maximum Temperature prediction.
    """
    return ClimateConvLSTMModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        kernel_size=kernel_size,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout
    )


if __name__ == "__main__":
    print("=" * 60)
    print("PYTORCH CLIMATE CONVLSTM ARCHITECTURE FORWARD PASS TEST")
    print("=" * 60)

    # Test parameters: Batch=2, Seq=7, C=25 (Rainfall), H=129 (Lat), W=135 (Lon)
    batch_size = 2
    seq_len = 7
    c_rain = 25
    h_lat = 129
    w_lon = 135

    # 1. Instantiate Rainfall ConvLSTM Model
    rf_convlstm = build_rainfall_convlstm_model(input_dim=c_rain, hidden_dims=[32, 32], num_layers=2, kernel_size=3)
    print("Rainfall ConvLSTM Architecture:")
    print(rf_convlstm)

    dummy_x_rain = torch.randn(batch_size, seq_len, c_rain, h_lat, w_lon)
    out_rain = rf_convlstm(dummy_x_rain)

    print(f"\nRainfall ConvLSTM Forward Pass Check:")
    print(f"  - Input Tensor Shape:  {dummy_x_rain.shape}  (B, T, C, H, W)")
    print(f"  - Output Target Shape: {out_rain.shape}  (Expected: ({batch_size}, {h_lat}, {w_lon}))")

    # 2. Test Temperature ConvLSTM Model & Autograd Backward Pass
    temp_convlstm = build_temperature_convlstm_model(input_dim=21, hidden_dims=32, num_layers=2)
    dummy_x_temp = torch.randn(batch_size, seq_len, 21, h_lat, w_lon)
    out_temp = temp_convlstm(dummy_x_temp)

    loss = out_temp.sum()
    loss.backward()

    print(f"\nTemperature ConvLSTM Forward & Backward Pass Check:")
    print(f"  - Output Shape: {out_temp.shape}")
    print(f"  - Autograd Backward Pass: PASSED")
    print("=" * 60)
