from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from transformers import WavLMModel


class WeightedLayerPooling(nn.Module):
    """
    Learnable feature aggregation across all hidden states of a speech SSL transformer.

    Computes a parameterized softmax distribution to dynamically weigh and collapse 
    the layer dimension into a unified structural representation.
    """

    def __init__(self, num_layers: int = 13):
        """
        Initializes the layer pooling module.

        Args:
            num_layers (int): Number of hidden states layers to aggregate.
        """
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, all_layer_outputs: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Blends hidden state layers using learnable importance weights.

        Args:
            all_layer_outputs (list or tuple): Hidden states from the SSL backbone. 
                Each tensor has shape [Batch, Time, Channels].

        Returns:
            torch.Tensor: Weighted pooled features of shape [Batch, Time, Channels].
        """
        stack = torch.stack(all_layer_outputs, dim=0)  # [Layers, Batch, Time, Channels]
        w = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return torch.sum(w * stack, dim=0)


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive Statistics Pooling (ASP) for capturing high-order acoustic benchmarks.

    Calculates channel-wise attentive mean and standard deviation over the temporal 
    axis to generate a fixed-size utterance-level representation.
    """

    def __init__(self, in_dim: int, bottleneck_dim: int = 128):
        """
        Initializes the attentive statistics pooling architecture.

        Args:
            in_dim (int): Input feature channel dimension.
            bottleneck_dim (int): Hidden dimension size for the attention mechanism.
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
        Extracts temporal statistics weighted by alignment scores.

        Args:
            x (torch.Tensor): Feature frames map tensor of shape [Batch, Channels, Time].

        Returns:
            torch.Tensor: Aggregated utterance embeddings of shape [Batch, Channels * 2].
        """
        weights = self.attention(x)
        mu = torch.sum(weights * x, dim=2)
        
        variance = torch.sum(weights * (x ** 2), dim=2) - (mu ** 2)
        std = torch.sqrt(torch.clamp(variance, min=1e-7))
        
        return torch.cat((mu, std), dim=1)


class GraphAttentionLayer(nn.Module):
    """
    Localized Graph Attention block tailored for sequence-based frame tokens.

    Combines regularized depthwise convolutions for localized sequential contexts 
    with Multi-Head Self-Attention acting as an explicit dense graph topology layer.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        """
        Initializes the temporal graph attention layer component.

        Args:
            in_dim (int): Dimension space of incoming frame nodes.
            out_dim (int): Output feature size mapping targets.
            heads (int): Number of multi-head attention components.
        """
        super().__init__()
        self.pos_conv = nn.Conv1d(
            in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim
        )
        self.gat = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=heads, batch_first=True
        )
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes node updates across sequence frames via localized self-attention steps.

        Args:
            x (torch.Tensor): Node context tensors of shape [Batch, Time, In_Dim].

        Returns:
            torch.Tensor: Structural node representations of shape [Batch, Time, Out_Dim].
        """
        residual = x
        
        # Inject localized structural dependencies across neighboring frames
        x_conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        
        # Execute attention mapping updates
        attn_out, _ = self.gat(x, x, x)
        x = self.norm(residual + attn_out)
        
        return F.gelu(self.fc(x))


class GNNWavLMSpoofDetector(nn.Module):
    """
    Dual-stream GNN anti-spoofing network built over self-supervised features.

    Leverages a fine-tuned WavLM encoder backbone combined with dual attentive 
    pooling pathways to isolate authentic acoustic properties from synthetic voice cues.
    """

    def __init__(self, freeze_ssl: bool = True, unfreeze_last_n: int = 2):
        """
        Initializes the dual-stream architecture setup.

        Args:
            freeze_ssl (bool): Flag to trigger parameter freezing routines.
            unfreeze_last_n (int): Count of trailing transformer blocks to keep trainable.
        """
        super().__init__()
        self.ssl_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.ssl_model.config.output_hidden_states = True

        if freeze_ssl:
            self.ssl_model.feature_extractor._freeze_parameters()
            for param in self.ssl_model.parameters():
                param.requires_grad = False
                
            if unfreeze_last_n > 0:
                for layer in self.ssl_model.encoder.layers[-unfreeze_last_n:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.layer_pooling = WeightedLayerPooling(num_layers=13)

        self.temporal_gat = nn.Sequential(
            GraphAttentionLayer(768, 768), 
            GraphAttentionLayer(768, 512)
        )

        self.asp_raw = AttentiveStatisticsPooling(768) 
        self.asp_gat = AttentiveStatisticsPooling(512)

        cls_in_dim = (768 * 2) + (512 * 2)

        self.classifier = nn.Sequential(
            nn.Linear(cls_in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts multi-layered acoustic contexts to predict speech authenticity.

        Args:
            x (torch.Tensor): Raw input speech waveforms of shape [Batch, Samples] 
                or flattened variations.

        Returns:
            torch.Tensor: Unnormalized classification logit values of shape [Batch].
        """
        if x.ndim > 2:
            x = x.view(-1, x.size(-1)) 
            
        outputs = self.ssl_model(x)
        all_layers = outputs.hidden_states

        # Extract normalized raw features
        raw_features = self.layer_pooling(all_layers)

        # Stream 1: Global statistical aggregation over raw acoustic signatures
        pooled_raw = self.asp_raw(raw_features.transpose(1, 2))

        # Stream 2: Deep temporal anomaly detection via Graph Attention paths
        gat_features = self.temporal_gat(raw_features)
        pooled_gat = self.asp_gat(gat_features.transpose(1, 2))

        # Merge global structural footprints with spatial node context matrices
        final_features = torch.cat([pooled_raw, pooled_gat], dim=1)

        return self.classifier(final_features).view(-1)