"""Shared MLP classification head."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def evaluate(model: nn.Module, dataset: TensorDataset, device: torch.device) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    correct = 0
    total = 0
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += labels.numel()
            loss_total += float(loss.item()) * labels.numel()
    return {"loss": loss_total / total, "accuracy": correct / total}
