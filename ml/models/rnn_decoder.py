from torch import nn


class RNNDecoder(nn.Module):
    """
    RNN-based sequential decoder for audio spoofing classification.

    Flattens spatial encoder feature maps along the temporal width axis,
    processes the sequence using an LSTM layer, and classifies the final hidden state.
    """

    def __init__(self, input_size: int, hidden_size: int = 128):
        """
        Initializes the RNNDecoder components.

        Args:
            input_size (int): Flattened feature dimension per time step (Channels * Height).
            hidden_size (int): Feature dimension size of the LSTM hidden state.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Decodes spatial-temporal feature tensors into classification logits.

        Args:
            x (torch.Tensor): Feature tensor from encoder of shape [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: Unnormalized classification logits of shape [Batch].
        """
        b, c, h, w = x.shape

        # Permute and flatten to treat the spatial width axis (W) as the timeline sequence
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, w, c * h)

        _, (hidden, _) = self.lstm(x)

        # Extract the hidden state from the last LSTM layer
        last_hidden = hidden[-1]
        logits = self.classifier(last_hidden)

        return logits.squeeze(1)
