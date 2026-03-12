import torch
import torch.nn as nn
import numpy as np
from torch.nn import init


class PoseEncoder(nn.Module):
    def __init__(self, out_channels=320, base_channels=4):
        super().__init__()

        hidden_channels = 4 * base_channels

        # Single-view video encoder:
        # [B,3,T,H,W] -> [B,4C,T,H,W] -> [B,4C,T,H/2,W/2]
        # -> [B,4C,T/4,H/8,W/8] -> [B,C_out,T/4,H/8,W/8]
        self.conv_layers = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=(1, 2, 2),
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv3d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

        self.final_proj = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )

        self.scale = nn.Parameter(torch.ones(1) * 2.0)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.conv_layers:
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2.0 / n))
                if m.bias is not None:
                    init.zeros_(m.bias)
        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)

    def _encode_single_view(self, x: torch.Tensor, strict_alignment: bool = True) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"`x` must be 5D [B,3,T,H,W], got shape {tuple(x.shape)}.")
        if x.shape[1] != 3:
            raise ValueError(f"`x` channel dim must be 3, got shape {tuple(x.shape)}.")
        _, _, t, h, w = x.shape
        if strict_alignment and (t % 4 != 0 or h % 8 != 0 or w % 8 != 0):
            raise ValueError(
                f"PoseEncoder expects T%4==0 and H/W%8==0 for exact Wan latent alignment, got shape {tuple(x.shape)}."
            )

        x = self.conv_layers(x)
        x = self.final_proj(x)
        return x * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # Backward-compatible path: [N,3,H,W] -> [N,C_out,H/8,W/8].
            x = x.unsqueeze(2)
            x = self._encode_single_view(x, strict_alignment=False)
            return x.squeeze(2)

        if x.ndim == 5:
            # Single-view video: [B,3,T,H,W].
            return self._encode_single_view(x)

        if x.ndim == 6:
            # Multi-view video: [B,V,3,T,H,W], encode each view independently.
            if x.shape[2] != 3:
                raise ValueError(f"`x` channel dim must be 3, got shape {tuple(x.shape)}.")
            encoded_views = [self._encode_single_view(x[:, v]) for v in range(x.shape[1])]
            return torch.stack(encoded_views, dim=1)

        raise ValueError(f"`x` must be 4D/5D/6D, got shape {tuple(x.shape)}.")
