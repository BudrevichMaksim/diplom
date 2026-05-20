import numpy as np
import torch
import torchaudio
import pyworld as pw
from torch import nn
from transformers import WavLMModel
from torchaudio.models import Conformer

from ml.models.prosody_encoder import ProsodyEncoder


class AttentiveLayerPooling(nn.Module):
    """Динамическое взвешивание 13 слоев WavLM"""

    def __init__(self, hidden_size=768):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.attn_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        stacked_states = torch.stack(hidden_states, dim=1)  # [B, L, T, D]

        # Смешиваем глобальные веса и локальное внимание к токенам
        global_w = torch.softmax(self.layer_weights, dim=0).view(1, -1, 1, 1)
        attn_w = torch.softmax(self.attn_proj(stacked_states), dim=1)

        weights = torch.softmax(attn_w + global_w, dim=1)

        return torch.sum(stacked_states * weights, dim=1)


class AttentivePooling1D(nn.Module):
    """Схлопывание временной оси T через Attention"""

    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        attn = torch.softmax(self.attention(x), dim=1)
        return torch.sum(attn * x, dim=1)


class MultiBranchWavLMSpoofDetector(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, n_last_layers=2):
        super().__init__()
        self.sample_rate = sample_rate

        # 1. SSL Branch (WavLM)
        self.ssl_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.ssl_model.config.output_hidden_states = True

        # Заморозка (кроме последних слоев для адаптации под задачу)
        for param in self.ssl_model.parameters():
            param.requires_grad = False
        if n_last_layers > 0:
            for layer in self.ssl_model.encoder.layers[-n_last_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.layer_pooling = AttentiveLayerPooling(768)
        self.ssl_proj = nn.Sequential(nn.Linear(768, 256), nn.Dropout(0.3))

        # 2. Prosody Encoder
        self.prosody_encoder = ProsodyEncoder(sample_rate=sample_rate)

        # 3. Mel Branch
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=320, n_mels=n_mels
        )
        self.mel_branch = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True),  # <--- Замена
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.InstanceNorm2d(128, affine=True),  # <--- Замена
            nn.GELU(),
        )
        self.mel_proj = nn.Linear(128 * (n_mels // 2), 256)

        # 4. Fusion & Conformer
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

    def forward(self, x):
        # x: [B, T_raw]

        # --- 1. SSL ---
        ssl_out = self.ssl_model(x).hidden_states
        ssl_feats = self.ssl_proj(self.layer_pooling(ssl_out))
        # print(ssl_feats)
        # --- 2. Prosody ---
        prosody_feats = self.prosody_encoder(x)
        # print(prosody_feats)

        # --- 3. Mel ---
        mel_spec = self.mel_transform(x)
        # clamp гарантирует, что значения никогда не опустятся ниже 1e-5
        mel = torch.log10(torch.clamp(mel_spec, min=1e-5)).unsqueeze(1)
        # print(mel)

        mel_feats = self.mel_branch(mel)
        # print(mel_feats)

        B, C, M, T = mel_feats.shape
        mel_feats = self.mel_proj(mel_feats.permute(0, 3, 1, 2).reshape(B, T, -1))
        # print(mel_feats)
        # --- 4. Fusion & Conformer ---
        min_t = min(ssl_feats.size(1), prosody_feats.size(1), mel_feats.size(1))
        fused = torch.cat(
            [ssl_feats[:, :min_t], prosody_feats[:, :min_t], mel_feats[:, :min_t]],
            dim=-1,
        )

        fused = self.fusion_proj(fused)
        # print(fused)
        # Conformer
        lengths = torch.full((x.size(0),), min_t, device=x.device)
        conf_out, _ = self.conformer(fused, lengths)

        # --- 5. Exit ---
        pooled = self.final_pooling(conf_out)
        # print(pooled)
        return self.classifier(pooled).view(-1)
