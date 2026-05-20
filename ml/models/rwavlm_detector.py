from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from transformers import WavLMModel


class WeightedLayerPooling(nn.Module):
    """
    Learnable hidden states optimization layer for WavLM features.

    Computes a parameterized softmax distribution across all transformer hidden 
    states to blend and collapse the layer dimension dynamically.
    """

    def __init__(self, num_layers: int = 13):
        """
        Initializes the learnable layer pooling module.

        Args:
            num_layers (int): Count of hidden state layers to aggregate.
        """
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, all_layer_outputs: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Aggregates layered hidden tokens into a weighted representation.

        Args:
            all_layer_outputs (list or tuple): List of layer hidden states, 
                each of shape [Batch, Time, Channels].

        Returns:
            torch.Tensor: Weighted pooled features of shape [Batch, Time, Channels].
        """
        stacked = torch.stack(all_layer_outputs, dim=0)  # [Layers, Batch, Time, Channels]
        weights = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        
        return torch.sum(weights * stacked, dim=0)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Mean and Standard Deviation Pooling (ASP) layer.

    Applies a 1D convolutional attention pooling mechanism across the temporal 
    axis to capture high-order global utterance statistics.
    """

    def __init__(self, in_dim: int, bottleneck_dim: int = 128):
        """
        Initializes the attentive statistics pooling block.

        Args:
            in_dim (int): Input feature channel dimension.
            bottleneck_dim (int): Hidden projection dimension for alignment weights.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates channel-wise attentive stats over temporal frames.

        Args:
            x (torch.Tensor): Feature matrices of shape [Batch, Channels, Time].

        Returns:
            torch.Tensor: Utterance level pooled representations of shape [Batch, Channels * 2].
        """
        weights = self.attention(x)
        mean = torch.sum(weights * x, dim=2)

        variance = torch.sum(weights * (x ** 2), dim=2) - (mean ** 2)
        std = torch.sqrt(torch.clamp(variance, min=1e-6))

        return torch.cat([mean, std], dim=1)


class LSTMWavLMSpoofDetector(nn.Module):
    """
    Recurrent self-supervised spoofing detector based on a WavLM backbone.

    Processes audio signals using downscaled fine-tuned transformer layers 
    interlocked with a bidirectional LSTM network and statistical pooling steps.
    """

    def __init__(
        self,
        freeze_ssl: bool = True,
        unfreeze_last_n: int = 0,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initializes the components and recurrent layers of the model.

        Args:
            freeze_ssl (bool): Flag to switch global model parameters freezing.
            unfreeze_last_n (int): Trailing transformer blocks count left trainable.
            lstm_hidden (int): Internal hidden states size for the LSTM layer paths.
            lstm_layers (int): Depth scaling factor for the stacked LSTM blocks.
            dropout (float): Dropout probability for intermediate LSTM sequences.
        """
        super().__init__()
        self.ssl_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.ssl_model.config.output_hidden_states = True
        ssl_dim = 768

        if freeze_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
            if unfreeze_last_n > 0:
                for layer in self.ssl_model.encoder.layers[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.layer_pooling = WeightedLayerPooling(num_layers=13)

        self.bottleneck = nn.Sequential(
            nn.Linear(ssl_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        lstm_out_dim = lstm_hidden * 2
        self.temporal_dropout = nn.Dropout(0.4)
        self.pooling = AttentiveStatisticsPooling(in_dim=lstm_out_dim)
        pooled_dim = lstm_out_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Applies Xavier initialization to linear projection layers."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts acoustic features and builds temporal graphs to predict speech tags.

        Args:
            x (torch.Tensor): Raw waveform sample values tensor of shape [Batch, Samples].

        Returns:
            torch.Tensor: Temperature scaled logit predictions of shape [Batch].
        """
        outputs = self.ssl_model(x)
        hidden_states = outputs.hidden_states

        # Extract features across all transformer blocks
        x = self.layer_pooling(hidden_states)  # [Batch, Time, 768]
        x = self.bottleneck(x)                # [Batch, Time, 256]

        # Process sequential structures via bidirectional dependencies
        x, _ = self.lstm(x)                    # [Batch, Time, Lstm_Hidden * 2]
        x = self.temporal_dropout(x)

        # Map temporal sequences onto unified statistics profiles
        x = x.transpose(1, 2)                  # [Batch, Lstm_Hidden * 2, Time]
        x = self.pooling(x)                    # [Batch, Lstm_Hidden * 4]

        # Scoring pipeline maps with scaling calibration updates
        logits = self.classifier(x)            # [Batch, 1]
        logits = logits / 4.0                  # Apply temperature logit calibration scaling

        return logits.view(-1)