import torch
import torch.nn as nn


class RNNHead(nn.Module):
    """Bidirectional GRU over T' time frames → frame-level species logits.

    Input:  (B, C, T')  — freq-pooled CNN features
    Output: (B, T', num_classes)
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        classifier_dropout: float = 0.3,
        num_classes: int = 234,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(gru_out_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T')
        Returns: (B, T', num_classes)
        """
        x = x.permute(0, 2, 1)       # (B, T', C)
        x, _ = self.gru(x)            # (B, T', gru_out_dim)
        x = self.classifier(x)        # (B, T', num_classes)
        return x
