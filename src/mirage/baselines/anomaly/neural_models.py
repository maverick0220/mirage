"""Compact, reproducible neural anomaly-detection baselines.

These are self-contained PyTorch implementations in the spirit of the KM-LagFormer
project's baseline pack (linear/MLP/LSTM/Transformer/MTGNN-style graph attention).
They learn a window-to-next-step predictor; the anomaly score is the prediction
MSE, calibrated on the validation split and evaluated with the same event/point
metrics as MIRAGE. Each model preserves the defining architecture of the named
baseline; for a publication claim, validate against the official repositories.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _mse_score(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target) ** 2).mean(dim=-1)


class LSTMPredictor(nn.Module):
    """Two-layer LSTM encoder -> linear head (OmniAnomaly-style backbone)."""

    def __init__(self, features: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.rnn = nn.LSTM(features, hidden, num_layers=layers, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, features)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.rnn(windows)
        return self.head(hidden[-1])


class TransformerPredictor(nn.Module):
    """Transformer encoder -> linear head (AnomalyTransformer/TranAD backbone)."""

    def __init__(self, features: int, d_model: int = 128, layers: int = 3, heads: int = 4):
        super().__init__()
        self.input = nn.Linear(features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, 0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, features)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input(windows))
        return self.head(encoded[:, -1])


class DLinearPredictor(nn.Module):
    """DLinear: time-wise linear decomposition (trend + residual)."""

    def __init__(self, features: int, window_size: int):
        super().__init__()
        self.trend = nn.Linear(window_size, 1)
        self.residual = nn.Linear(window_size, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        transposed = windows.transpose(1, 2)  # [B, D, W]
        return (self.trend(transposed) + self.residual(transposed)).squeeze(-1)


class GDNPredictor(nn.Module):
    """GDN-style: temporal CNN + learnable node-embedding graph attention."""

    def __init__(self, features: int, hidden: int = 128):
        super().__init__()
        self.temporal = nn.Conv1d(features, hidden, kernel_size=3, padding=1)
        self.node_embedding = nn.Parameter(torch.randn(features, 16) / math.sqrt(16))
        self.fuse = nn.Linear(hidden + features, hidden)
        self.head = nn.Linear(hidden, features)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        temporal = torch.relu(self.temporal(windows.transpose(1, 2)))  # [B, H, W]
        adjacency = torch.softmax(self.node_embedding @ self.node_embedding.T, dim=-1)
        node_signal = windows[:, -1] @ adjacency  # [B, D]
        state = torch.relu(self.fuse(torch.cat([temporal.mean(-1), node_signal], dim=-1)))
        return self.head(state)


class OmniAnomalyPredictor(nn.Module):
    """Compact VAE reconstruction baseline (OmniAnomaly-style stochastic RNN)."""

    def __init__(self, features: int, hidden: int = 128, latent: int = 8):
        super().__init__()
        self.encoder = nn.LSTM(features, hidden, batch_first=True)
        self.mean = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.decoder = nn.LSTM(latent, hidden, batch_first=True)
        self.reconstruct = nn.Linear(hidden, features)
        self.latent = latent

    def forward(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.encoder(windows)
        state = hidden[-1]
        mean, logvar = self.mean(state), self.logvar(state)
        sample = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        decoded, _ = self.decoder(sample.unsqueeze(1).repeat(1, windows.shape[1], 1))
        reconstruction = self.reconstruct(decoded)
        return reconstruction, mean, logvar

    def score_batch(self, windows: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reconstruction, mean, logvar = self(windows)
        last = reconstruction[:, -1]
        mse = ((last - target) ** 2).mean(dim=-1)
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1)
        return mse + 0.1 * kl


class MTADGATPredictor(nn.Module):
    """MTAD-GAT-style: feature attention + temporal attention -> linear head."""

    def __init__(self, features: int, window_size: int, hidden: int = 128):
        super().__init__()
        self.feature_attn = nn.MultiheadAttention(features, num_heads=2, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(window_size, num_heads=2, batch_first=True)
        self.head = nn.Linear(features * window_size, features)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # Feature attention over variables.
        features, _ = self.feature_attn(windows, windows, windows)  # [B, W, D]
        # Temporal attention over time.
        temporal, _ = self.temporal_attn(
            features.transpose(1, 2), features.transpose(1, 2), features.transpose(1, 2)
        )
        return self.head(temporal.transpose(1, 2).reshape(len(windows), -1))


PREDICTOR_REGISTRY = {
    "lstm": LSTMPredictor,
    "transformer": TransformerPredictor,
    "anomaly_transformer": TransformerPredictor,
    "tranad": TransformerPredictor,
    "dlinear": DLinearPredictor,
    "gdn": GDNPredictor,
    "omnianomaly": OmniAnomalyPredictor,
    "mtad_gat": MTADGATPredictor,
}


def build_neural_predictor(name: str, features: int, window_size: int) -> nn.Module:
    normalized = name.lower().replace("-", "").replace("_", "")
    if normalized == "dlinear":
        return DLinearPredictor(features, window_size)
    if normalized == "mtadgat":
        return MTADGATPredictor(features, window_size)
    if normalized == "lstm":
        return LSTMPredictor(features)
    if normalized in {"transformer", "anomalytransformer", "tranad"}:
        return TransformerPredictor(features)
    if normalized == "gdn":
        return GDNPredictor(features)
    if normalized == "omnianomaly":
        return OmniAnomalyPredictor(features)
    raise KeyError(f"Unknown neural baseline: {name}")
