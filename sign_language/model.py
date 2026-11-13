"""LSTM classifier for sequences of MediaPipe landmarks."""

import torch
from torch import nn

from sign_language.landmarks import FEATURE_SIZE


class SignSequenceClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: int = FEATURE_SIZE,
        hidden_size: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.recurrent(sequence)
        return self.classifier(outputs[:, -1, :])
