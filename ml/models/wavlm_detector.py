from typing import List, Tuple, Union
import torch
import torch.nn as nn
from transformers import WavLMModel


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Mean and Standard Deviation Pooling (ASP).

    Extracts temporal statistics (weighted mean and standard deviation) from 
    frame-level features using a 1D convolutional attention mechanism.
    """

    def __init__(self, in_dim: int, bottleneck_dim: int = 128):
        """
        Initializes the attentive statistics pooling layer.

        Args:
            in_dim (int): Input feature channel dimension.
            bottleneck_dim (int): Hidden projection dimension for attention calculation.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes weighted temporal statistics over the sequence frames.

        Args:
            x (torch.Tensor): Feature frames map of shape [Batch, Channels, Time].

        Returns:
            torch.Tensor: Pooled utterance representation of shape [Batch, Channels * 2].
        """
        weights = self.attention(x)
        mu = torch.sum(weights * x, dim=2)
        
        # Calculate variance safely to prevent negative numbers before sqrt
        variance = torch.sum(weights * (x ** 2), dim=2) - (mu ** 2)
        std = torch.sqrt(torch.clamp(variance, min=1e-5))
        
        return torch.cat((mu, std), dim=1)


class WavLMSpoofDetector(nn.Module):
    """
    Baseline WavLM-based speech anti-spoofing detector.

    Leverages raw last hidden states from a pre-trained SSL backbone, 
    aggregates temporal dynamics via ASP, and scores final logs.
    """

    def __init__(self, freeze_ssl: bool = True):
        """
        Initializes the baseline spoof detector pipeline.

        Args:
            freeze_ssl (bool): Flag to trigger parameter freezing routines.
        """
        super().__init__()
        self.ssl_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")

        if freeze_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False

        ssl_out_dim = 768
        self.pooling = AttentiveStatisticsPooling(ssl_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(ssl_out_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts raw SSL representations and calculates pooling scores.

        Args:
            x (torch.Tensor): Input audio waveforms of shape [Batch, Samples] 
                or flattened variations.

        Returns:
            torch.Tensor: Unnormalized output logit values of shape [Batch].
        """
        if x.ndim > 2:
            x = x.view(-1, x.size(-1))

        outputs = self.ssl_model(x)

        x = outputs.last_hidden_state.transpose(1, 2)  # [Batch, 768, Time]
        x = self.pooling(x)                            # [Batch, 1536]
        
        return self.classifier(x).view(-1)