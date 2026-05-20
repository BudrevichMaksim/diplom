from torch import nn

from ml.models.cnn_encoder import CNNEncoder
from ml.models.rnn_decoder import RNNDecoder


class RCNNSpoofDetector(nn.Module):
    """
    Recurrent Convolutional Neural Network (RCNN) for audio spoofing detection.

    Combines a CNN-based front-end encoder to extract spatial/spectral features
    with an RNN-based back-end decoder to capture sequential temporal dependencies.
    """

    def __init__(self, input_size: int = 2048):
        """
        Initializes the RCNNSpoofDetector components.

        Args:
            input_size (int): Expected feature dimension size transferred from the
                encoder to the decoder.
        """
        super().__init__()
        self.encoder = CNNEncoder()
        self.decoder = RNNDecoder(input_size=input_size)

    def forward(self, x):
        """
        Processes the input features through the spatial encoder and temporal decoder.

        Args:
            x (torch.Tensor): Input spectrograms or front-end raw feature representations.

        Returns:
            torch.Tensor: Flattened 1D tensor of unnormalized network logits.
        """
        features = self.encoder(x)
        return self.decoder(features).view(-1)
