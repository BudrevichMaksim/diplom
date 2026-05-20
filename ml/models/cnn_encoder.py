import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard 2D convolutional block with normalization, activation, pooling, and dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.25):
        """
        Initializes the convolutional structural block.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution.
            dropout (float): 2D spatial dropout probability rate.
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Spatial dropout drops entire channels rather than individual elements
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor through the sequence of layers.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: Downsampled and filtered feature map tensor.
        """
        return self.block(x)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for dynamic channel-wise feature recalibration.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Initializes the Squeeze-and-Excitation framework block.

        Args:
            channels (int): Number of input feature map channels.
            reduction (int): Reduction ratio for the internal bottleneck layer bottleneck.
        """
        super().__init__()
        bottleneck_channels = max(1, channels // reduction)

        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies feature map scaling based on computed channel relationships.

        Args:
            x (torch.Tensor): Input map layer tensor of shape [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: Scaled excitation tensor copy of identical shape layout.
        """
        b, c, _, _ = x.size()
        
        # Squeeze: Global Average Pooling across spatial dimensions (Height and Width)
        y = x.mean(dim=(2, 3))
        
        # Excitation: Compute adaptive channel scale weights
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)


class CNNEncoder(nn.Module):
    """
    Feature mapping block executing front-end structural encoding for audio spectrograms.
    """

    def __init__(self):
        """Initializes the multi-layer deep feature block pipeline sequence."""
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            SEBlock(64),
            ConvBlock(64, 128),
            SEBlock(128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts complex latent spatial feature mappings from incoming maps.

        Args:
            x (torch.Tensor): Raw spectrogram input matrix tensor of shape 
                [Batch, 1, Frequency_Bins, Time_Frames].

        Returns:
            torch.Tensor: Deep spatial feature representation embedding map tensor.
        """
        return self.features(x)