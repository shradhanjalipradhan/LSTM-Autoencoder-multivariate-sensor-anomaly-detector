"""
LSTM Autoencoder for multivariate time-series anomaly detection.

Architecture
------------
Encoder : stacked LSTM  →  bottleneck hidden state (latent vector)
Decoder : bottleneck repeated seq_len times  →  stacked LSTM  →  Linear  →  reconstruction
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM Autoencoder.

    Parameters
    ----------
    n_features : int
        Number of input channels (4 for our sensor set).
    hidden_dim : int
        LSTM hidden-state dimension.
    latent_dim : int
        Compressed representation dimension (bottleneck).
    n_layers : int
        Number of stacked LSTM layers in encoder and decoder.
    seq_len : int
        Length of the sliding-window sequence fed to the model.
    dropout : float
        Dropout probability applied between LSTM layers.
    """

    def __init__(
        self,
        n_features: int = 4,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_layers: int = 2,
        seq_len: int = 50,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        # --- Encoder ---
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # --- Decoder ---
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) → latent: (batch, latent_dim)"""
        _, (hidden, _) = self.encoder_lstm(x)
        # Take the last layer's hidden state
        latent = self.encoder_fc(hidden[-1])
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: (batch, latent_dim) → recon: (batch, seq_len, n_features)"""
        expanded = self.decoder_fc(latent)                        # (B, hidden_dim)
        repeated = expanded.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, H)
        decoded, _ = self.decoder_lstm(repeated)
        recon = self.output_layer(decoded)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)

    # ------------------------------------------------------------------
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared reconstruction error.

        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        errors : (batch,)  — MSE per window
        """
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
            errors = ((x - recon) ** 2).mean(dim=(1, 2))
        return errors
