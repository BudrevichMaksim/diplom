import torch
import torch.nn as nn
import torchaudio.functional as F


class ProsodyEncoder(nn.Module):
    """
    Prosody Feature Encoder for extraction of speech intonation trajectories.

    Computes frame-level pitch (F0), root-mean-square energy, and delta pitch
    trajectories on the GPU, regularizing them through a 1D CNN and a
    bidirectional LSTM layer into a joint prosodic space.
    """

    def __init__(self, sample_rate: int = 16000, hidden_dim: int = 128):
        """
        Initializes the prosody tracking architecture components.

        Args:
            sample_rate (int): Audio processing sampling rate frequency benchmark.
            hidden_dim (int): Internal hidden states dimension for the recurrent block.
                Outputs a channel space of size hidden_dim * 2 due to bidirectionality.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = int(sample_rate * 0.02)  # 320 samples (20ms frame shift)
        self.win_length = int(sample_rate * 0.03)  # 480 samples (30ms window size)

        # 1. Instance Normalization to bridge different metric scales (Hz vs RMS)
        self.input_norm = nn.InstanceNorm1d(3, affine=True)

        # 2. Convolutional path for extracting localized dynamic patterns
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        # 3. Recurrent path capturing structural global temporal intonations
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts temporal prosody parameters and builds aggregated tracking states.

        Args:
            x (torch.Tensor): Raw input speech waveforms on the target device
                of shape [Batch, Samples].

        Returns:
            torch.Tensor: Continuous prosodic feature representations
                of shape [Batch, Time_frames, Hidden_dim * 2].
        """
        # 1. GPU-accelerated Pitch (F0) estimation via YIN algorithm
        pitch = F.detect_pitch_frequency(
            x, self.sample_rate, frame_time=0.02, win_length=30
        )  # [Batch, Time_pitch]
        pitch = torch.nan_to_num(pitch, nan=0.0)

        # 2. Multi-channel Root-Mean-Square (RMS) Energy computation
        energy = (
            x.unfold(-1, self.win_length, self.hop_length)
            .pow(2)
            .mean(-1)
            .clamp(min=1e-5)
            .sqrt()
        )
        # [Batch, Time_energy]

        # 3. Synchronize sequence lengths to avoid framing rounding mismatches
        min_len = min(pitch.size(1), energy.size(1))
        pitch = pitch[:, :min_len]
        energy = energy[:, :min_len]

        # 4. Human-centric logarithmic transformation mapping
        pitch = torch.log1p(pitch)
        energy = torch.log1p(energy)

        # 5. Compute Delta Pitch (first-order derivative for tone transition speed)
        delta_pitch = torch.zeros_like(pitch)
        delta_pitch[:, 1:] = pitch[:, 1:] - pitch[:, :-1]

        # 6. Build the unified feature matrix
        features = torch.stack(
            [pitch, energy, delta_pitch], dim=1
        )  # [Batch, 3, Min_Time]

        # 7. Physical scale normalization and neural map transitions
        features = self.input_norm(features)
        features = torch.nan_to_num(features, nan=0.0)

        # Pass through spatial-temporal network pipelines
        x_out = self.cnn(features)  # [Batch, 128, Min_Time]

        x_out = x_out.transpose(1, 2)  # [Batch, Min_Time, 128]
        x_out, _ = self.lstm(x_out)  # [Batch, Min_Time, Hidden_dim * 2]

        return x_out
