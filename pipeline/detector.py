"""
Anomaly detector built on top of the LSTM Autoencoder.

Workflow
--------
1. Train the autoencoder on *normal* (label==0) training windows.
2. Compute reconstruction error on the validation set to set the threshold
   at (mean + k * std) of the *normal* error distribution.
3. Predict anomalies on any new windowed tensor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_autoencoder import LSTMAutoencoder


class AnomalyDetector:
    """Train, threshold, and predict with an LSTMAutoencoder.

    Parameters
    ----------
    n_features   : number of sensor channels
    hidden_dim   : LSTM hidden size
    latent_dim   : bottleneck size
    n_layers     : number of stacked LSTM layers
    seq_len      : sliding-window length
    dropout      : LSTM dropout
    lr           : learning-rate for Adam
    batch_size   : mini-batch size
    n_epochs     : maximum training epochs
    patience     : early-stopping patience (epochs without val improvement)
    threshold_k  : number of std-devs above normal mean for the error threshold
    device       : 'cpu' | 'cuda'  (auto-detected if None)
    """

    def __init__(
        self,
        n_features: int = 4,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_layers: int = 2,
        seq_len: int = 50,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 128,
        n_epochs: int = 50,
        patience: int = 7,
        threshold_k: float = 3.0,
        device: str | None = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.patience = patience
        self.threshold_k = threshold_k
        self.threshold: float | None = None

        self.model = LSTMAutoencoder(
            n_features=n_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
            seq_len=seq_len,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: np.ndarray,
        X_val: torch.Tensor,
        y_val: np.ndarray,
    ) -> list[float]:
        """Train on *normal* windows only; use val loss for early stopping.

        Returns
        -------
        train_losses : list of per-epoch average training loss
        """
        # Keep only normal windows for training
        normal_mask_train = y_train == 0
        X_train_normal = X_train[normal_mask_train]

        normal_mask_val = y_val == 0
        X_val_normal = X_val[normal_mask_val]

        print(
            f"[Detector] Training on {len(X_train_normal)} normal windows "
            f"(val: {len(X_val_normal)} normal windows)"
        )

        train_loader = DataLoader(
            TensorDataset(X_train_normal),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_normal),
            batch_size=self.batch_size,
            shuffle=False,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        train_losses: list[float] = []

        for epoch in range(1, self.n_epochs + 1):
            # --- Train ---
            self.model.train()
            epoch_loss = 0.0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.criterion(recon, batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(X_train_normal)
            train_losses.append(epoch_loss)

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(self.device)
                    recon = self.model(batch)
                    val_loss += self.criterion(recon, batch).item() * len(batch)
            val_loss /= max(len(X_val_normal), 1)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{self.n_epochs}  "
                    f"train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}"
                )

            # Early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Set threshold from val-normal reconstruction errors
        self._fit_threshold(X_val_normal)
        return train_losses

    # ------------------------------------------------------------------
    def _fit_threshold(self, X_normal: torch.Tensor) -> None:
        errors = self._compute_errors(X_normal)
        mu, sigma = errors.mean(), errors.std()
        self.threshold = float(mu + self.threshold_k * sigma)
        print(
            f"[Detector] Threshold = {self.threshold:.6f}  "
            f"(mu={mu:.6f}, sigma={sigma:.6f}, k={self.threshold_k})"
        )

    # ------------------------------------------------------------------
    def _compute_errors(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        all_errors: list[np.ndarray] = []
        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size)
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                errs = self.model.reconstruction_error(batch).cpu().numpy()
                all_errors.append(errs)
        return np.concatenate(all_errors)

    # ------------------------------------------------------------------
    def predict(self, X: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Return per-window reconstruction errors and binary anomaly flags.

        Returns
        -------
        errors    : (N,) float32 array
        anomalies : (N,) int32 array  (1 = anomaly)
        """
        if self.threshold is None:
            raise RuntimeError("Call fit() before predict().")
        errors = self._compute_errors(X)
        anomalies = (errors > self.threshold).astype(np.int32)
        return errors, anomalies

    # ------------------------------------------------------------------
    def evaluate(
        self, X: torch.Tensor, y_true: np.ndarray, split_name: str = "test"
    ) -> dict[str, float]:
        """Compute precision, recall, F1, accuracy."""
        errors, preds = self.predict(X)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        accuracy = (tp + tn) / max(len(y_true), 1)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }
        print(
            f"[Detector] {split_name} -> "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        return metrics
