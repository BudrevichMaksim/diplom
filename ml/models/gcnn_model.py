import torch
from torch import nn

from ml.models.cnn_encoder import CNNEncoder
from ml.models.gat_decoder import GATDecoder


class GCNNSpoofDetector(nn.Module):
    """
    Graph-Convolutional Neural Network (GCNN) for speech anti-spoofing detection.

    Integrates a deep 2D CNN encoder for localized spectral feature extraction 
    with a windowed Graph Attention Network (GAT) decoder to capture structural 
    temporal anomalies across frame tokens.
    """

    def __init__(self, input_size: int = 2048):
        """
        Initializes the composite GCNN spoofing detector topology.

        Args:
            input_size (int): Flattened channel-frequency spatial footprint dimension 
                expected by the graph-based decoder stage.
        """
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = GATDecoder(input_size=input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes spatial spectral maps and decodes graph relationships to score authenticity.

        Args:
            x (torch.Tensor): Input acoustic spectrograms or raw feature matrices 
                of shape [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: Unnormalized output logit classification values of shape [Batch].
        """
        # Step 1: Extract continuous structural 2D spatial feature maps
        features = self.encoder(x)  # [Batch, Out_Channels, Height_prime, Width_prime]
        
        # Step 2: Formulate sliding-window graph nodes and calculate attention states
        return self.decoder(features).view(-1)