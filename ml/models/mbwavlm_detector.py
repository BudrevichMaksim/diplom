from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torchaudio.models import Conformer
from transformers import WavLMModel

from ml.models.prosody_encoder import ProsodyEncoder


class AttentiveLayerPooling(nn.Module):
    """
    Dynamic attention-driven layer aggregation for WavLM hidden states.

    Blends global learnable layer weights with localized token-level attention 
    to dynamically collapse the transformer layer dimension.
    """

    def __init__(self, hidden_size: int = 768):
        """
        Initializes the attentive layer pooling module.

        Args:
            hidden_size (int): Expected hidden dimension of the SSL backbone models.
        """
        super().__init__()
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.attn_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """
        Aggregates layered hidden tokens via combined localized and global attention.

        Args:
            hidden_states (list or tuple): Hidden representations of shape [Batch, Time, Dimension].

        Returns:
            torch.Tensor: Weighted context representations of shape [Batch, Time, Dimension].
        """
        stacked_states = torch.stack(hidden_states, dim=1)  # [Batch, Layers, Time, Dimension]

        global_w = torch.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)
        attn_w = torch.softmax(self.attn_proj(stacked_states), dim=1)

        weights = torch.softmax(attn_w + global_w, dim=1)
        return torch.sum(stacked_states * weights, dim=1)


class AttentivePooling1D(nn.Module):
    """
    Temporal attention pooling layer.

    Collapses the sequential time axis into a fixed-size utterance-level embedding 
    using a localized linear scoring mechanism.
    """

    def __init__(self, dim: int):
        """
        Initializes the 1D attentive pooling layer.

        Args:
            dim (int): Vector feature size of the incoming sequential tokens.
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2), 
            nn.Tanh(), 
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses time frames into global weighted embeddings.

        Args:
            x (torch.Tensor): Sequential features matrix of shape [Batch, Time, Dimension].

        Returns:
            torch.Tensor: Pooled utterance representations of shape [Batch, Dimension].
        """
        attn = torch.softmax(self.attention(x), dim=1)
        return torch.sum(attn * x, dim=1)


class MultiBranchWavLMSpoofDetector(nn.Module):
    """
    Multi-branch spoofing detector combining SSL, Prosody, and Mel-spectrogram models.

    Fuses structural speech representations, acoustic frequency spectrogram maps, and 
    prosodic trajectories through a localized Conformer encoder block.
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, n_last_layers: int = 2):
        """
        Initializes the multi-branch spoof detector architecture.

        Args:
            sample_rate (int): Audio processing sampling rate frequency benchmark.
            n_mels (int): Target frequency bin count for structural Mel calculations.
            n_last_layers (int): Count of trailing WavLM transformer layers to unfreeze.
        """
        super().__init__()
        self.sample_rate = sample_rate

        # 1. SSL Branch Setup
        self.ssl_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.ssl_model.config.output_hidden_states = True

        for param in self.ssl_model.parameters():
            param.requires_grad = False
            
        if n_last_layers > 0:
            for layer in self.ssl_model.encoder.layers[-n_last_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.layer_pooling = AttentiveLayerPooling(768)
        self.ssl_proj = nn.Sequential(
            nn.Linear(768, 256), 
            nn.Dropout(0.3)
        )

        # 2. Prosody Tracking Setup
        self.prosody_encoder = ProsodyEncoder(sample_rate=sample_rate)

        # 3. Mel Spectrogram Processing Pipeline
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=320, n_mels=n_mels
        )
        self.mel_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.GELU(),
        )
        self.mel_proj = nn.Linear(128 * (n_mels // 2), 256)

        # 4. Multi-stream Fusion & Conformer Integration
        self.fusion_proj = nn.Linear(256 + 256 + 256, 512)
        self.conformer = Conformer(
            input_dim=512,
            num_heads=8,
            ffn_dim=1024,
            num_layers=4,
            depthwise_conv_kernel_size=31,
        )
        self.final_pooling = AttentivePooling1D(512)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes complete multi-branch fusion forward pass to score speech validity.

        Args:
            x (torch.Tensor): Raw target input acoustic waveforms of shape [Batch, Samples].

        Returns:
            torch.Tensor: Unnormalized output logit prediction tensor mapping of shape [Batch].
        """
        # --- 1. SSL Branch processing ---
        ssl_out = self.ssl_model(x).hidden_states
        ssl_feats = self.ssl_proj(self.layer_pooling(ssl_out))  # [Batch, Time_ssl, 256]

        # --- 2. Prosody Branch processing ---
        prosody_feats = self.prosody_encoder(x)                # [Batch, Time_prosody, 256]

        # --- 3. Mel Branch processing ---
        mel_spec = self.mel_transform(x)
        mel = torch.log10(torch.clamp(mel_spec, min=1e-5)).unsqueeze(1)  # [Batch, 1, Mels, Time_raw]

        mel_feats = self.mel_branch(mel)                        # [Batch, 128, Mels // 2, Time_mel]
        b, c, m, t = mel_feats.shape
        
        mel_feats = mel_feats.permute(0, 3, 1, 2).reshape(b, t, -1)
        mel_feats = self.mel_proj(mel_feats)                    # [Batch, Time_mel, 256]

        # --- 4. Stream Synchronization & Fusion ---
        min_t = min(ssl_feats.size(1), prosody_feats.size(1), mel_feats.size(1))
        fused = torch.cat(
            [ssl_feats[:, :min_t], prosody_feats[:, :min_t], mel_feats[:, :min_t]],
            dim=-1,
        )  # [Batch, Min_Time, 768]

        fused = self.fusion_proj(fused)  # [Batch, Min_Time, 512]

        # --- 5. Feature Refinement via Conformer ---
        lengths = torch.full((x.size(0),), min_t, device=x.device)
        conf_out, _ = self.conformer(fused, lengths)  # [Batch, Min_Time, 512]

        # --- 6. Readout and Classification ---
        pooled = self.final_pooling(conf_out)  # [Batch, 512]
        return self.classifier(pooled).view(-1)