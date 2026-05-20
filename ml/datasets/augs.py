import random
import torch
import torch.nn.functional as F
import torchaudio
from typing import Tuple


class RawAudioAugmentor:
    """
    Online data augmentation pipeline for raw 1D/2D audio waveforms.

    Applies stochastic transformations including noise injection, volume scaling,
    low-pass filtering, signal clipping, temporal masking, and synthetic reverberation
    to improve classifier robustness against codec and acoustic variations.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_prob: float = 0.4,
        gain_prob: float = 0.4,
        lowpass_prob: float = 0.4,
        reverb_prob: float = 0.25,
        mask_prob: float = 0.2,
        clip_prob: float = 0.15,
    ):
        """
        Initializes the augmentor execution probabilities.

        Args:
            sample_rate (int): Sampling rate of the processed audio signals.
            noise_prob (float): Probability of applying additive Gaussian noise.
            gain_prob (float): Probability of applying random volume gain adjustments.
            lowpass_prob (float): Probability of applying a biquad low-pass filter.
            reverb_prob (float): Probability of applying convolution-based reverb.
            mask_prob (float): Probability of applying partial time-domain attenuation.
            clip_prob (float): Probability of applying soft-clipping distortion.
        """
        self.sample_rate = sample_rate
        self.noise_prob = noise_prob
        self.gain_prob = gain_prob
        self.lowpass_prob = lowpass_prob
        self.reverb_prob = reverb_prob
        self.mask_prob = mask_prob
        self.clip_prob = clip_prob

    def add_noise(
        self, waveform: torch.Tensor, snr_range: Tuple[float, float] = (10.0, 25.0)
    ) -> torch.Tensor:
        """
        Injects additive white Gaussian noise calibrated to a specific SNR target range.

        Args:
            waveform (torch.Tensor): Audio tensor of shape [Channels, Time].
            snr_range (Tuple[float, float]): Minimum and maximum target Signal-to-Noise ratio in dB.

        Returns:
            torch.Tensor: Noisy audio waveform.
        """
        snr = random.uniform(*snr_range)
        noise = torch.randn_like(waveform)

        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()

        scale = torch.sqrt(signal_power / (10 ** (snr / 10) * (noise_power + 1e-8)))
        return waveform + scale * noise

    def random_gain(
        self, waveform: torch.Tensor, gain_db: Tuple[float, float] = (-3.0, 3.0)
    ) -> torch.Tensor:
        """
        Applies random volume adjustments within a specified decibel boundary.

        Args:
            waveform (torch.Tensor): Audio tensor of shape [Channels, Time].
            gain_db (Tuple[float, float]): Gain range boundaries in decibels.

        Returns:
            torch.Tensor: Amplitude-scaled audio waveform.
        """
        gain = random.uniform(*gain_db)
        return waveform * (10 ** (gain / 20))

    def lowpass(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applies a biquad low-pass filter using a randomized frequency cutoff point.

        Args:
            waveform (torch.Tensor): Audio tensor of shape [Channels, Time].

        Returns:
            torch.Tensor: High-frequency degraded audio waveform.
        """
        cutoff = random.randint(3000, 7000)
        return torchaudio.functional.lowpass_biquad(
            waveform,
            sample_rate=self.sample_rate,
            cutoff_freq=cutoff,
        )

    def soft_clip(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Simulates microphone preamp or audio codec saturation using a hyperbolic tangent.

        Args:
            waveform (torch.Tensor): Audio tensor of shape [Channels, Time].

        Returns:
            torch.Tensor: Soft-clipped distorted audio waveform.
        """
        drive = random.uniform(1.2, 2.5)
        return torch.tanh(waveform * drive)

    def time_mask(
        self, waveform: torch.Tensor, max_duration: float = 0.12
    ) -> torch.Tensor:
        """
        Attenuates a contiguous temporal block to simulate transient drops or channel fade.

        Args:
            waveform (torch.Tensor): Audio tensor of shape [Channels, Time].
            max_duration (float): Maximum window scale allocated for masking in seconds.

        Returns:
            torch.Tensor: Temporally masked audio waveform.
        """
        length = waveform.shape[1]
        mask_len = int(random.uniform(0.03, max_duration) * self.sample_rate)

        if length <= mask_len:
            return waveform

        start = random.randint(0, length - mask_len)
        attenuation = random.uniform(0.05, 0.4)

        waveform[:, start : start + mask_len] *= attenuation
        return waveform

    def simple_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Simulates room acoustics via an explicit multi-tap delay line 1D convolution.

        Args:
            waveform (torch.Tensor): Audio tensor of shape [Channels, Time].

        Returns:
            torch.Tensor: Reverberated audio waveform output.
        """
        delay = random.randint(120, 400)
        decay = random.uniform(0.15, 0.35)

        impulse = torch.zeros(delay * 3)
        impulse[0] = 1.0
        impulse[delay] = decay
        impulse[delay * 2] = decay * 0.5

        impulse = impulse.to(waveform.device)

        # Expand dims to process via channel-agnostic 1D convolution
        out = F.conv1d(
            waveform.unsqueeze(1),
            impulse.view(1, 1, -1),
            padding=impulse.shape[0] // 2,
        ).squeeze(1)

        return out

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Sequentially executes the random augmentation pipeline transformations.

        Args:
            waveform (torch.Tensor): Input raw audio tensor of shape [Channels, Time].

        Returns:
            torch.Tensor: Augmented, length-conformed, and clamped audio waveform.
        """
        waveform = waveform.clone()

        if random.random() < self.gain_prob:
            waveform = self.random_gain(waveform)

        if random.random() < self.lowpass_prob:
            waveform = self.lowpass(waveform)

        if random.random() < self.noise_prob:
            waveform = self.add_noise(waveform)

        if random.random() < self.mask_prob:
            waveform = self.time_mask(waveform)

        if random.random() < self.clip_prob:
            waveform = self.soft_clip(waveform)

        if random.random() < self.reverb_prob:
            waveform = self.simple_reverb(waveform)

        waveform = torch.clamp(waveform, -1.0, 1.0)

        # Force structural frame dimensions length compliance (4 seconds)
        target_len = self.sample_rate * 4
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        elif waveform.shape[1] < target_len:
            pad = target_len - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad))

        return waveform
