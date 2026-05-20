import torch
import torch.nn as nn
import torchaudio.transforms as T


class MelSpectrogramExtractor(nn.Module):
    """
    On-the-fly Mel-Spectrogram feature extractor module.

    Stochastically or deterministically converts raw time-domain waveforms into
    log-mel power spectrograms (dB scale). Designed to run dynamically on either
    CPU or GPU as a front-end module directly inside a neural network pipeline.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
    ):
        """
        Initializes the MelSpectrogramExtractor transform blocks.

        Args:
            sample_rate (int): Sampling rate of the input raw audio waveform.
            n_fft (int): Size of FFT; determines frequency resolution boundaries.
            hop_length (int): Distance between successive window frames.
            n_mels (int): Number of Mel filterbanks to project the spectrum onto.
        """
        super().__init__()
        self.sample_rate = sample_rate

        self.transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # Converts linear scale power spectrogram to decibel log compression units
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extracts log-mel spectrogram features from an input audio waveform tensor.

        Args:
            waveform (torch.Tensor): Waveform tensor of shape [Batch, Time]
                or [Batch, 1, Time].

        Returns:
            torch.Tensor: Log-mel power spectrogram tensor of shape
                [Batch, 1, n_mels, Frames], perfectly formatted for 2D CNN layers.
        """
        # Inject an explicit channel dimension if missing to ensure 2D CNN compliance
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)  # [Batch, Time] -> [Batch, 1, Time]

        # Compute mel-scale frequency power distribution
        spec = self.transform(waveform)

        # Apply log compression to simulate perceptual human loudness scales
        spec_db = self.amplitude_to_db(spec)

        return spec_db
