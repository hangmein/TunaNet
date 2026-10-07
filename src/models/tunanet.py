"""
TunaNet - mạng CNN tự thiết kế phân loại chất lượng thịt cá ngừ.

    Ảnh Lab + Mask (B, 4, H, W)
       ├─> Nhánh kết cấu (CNN + MaskedGAP) ─> 256 ┐
       └─> Nhánh màu (thống kê Lab) ────────>  64 ┤
                                                  ↓
                 Concat 320 -> FC 128 -> ReLU -> Dropout -> FC num_classes

Các cờ use_color / use_se dùng cho thí nghiệm ablation
(chứng minh từng thành phần có đóng góp).
"""
import torch
import torch.nn as nn

from .blocks import init_weights
from .texture_branch import TextureBranch
from .color_branch import ColorBranch


class TunaNet(nn.Module):
    def __init__(self, num_classes=3, use_color=True, use_se=True,
                 widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2), dropout=0.5):
        super().__init__()
        self.texture = TextureBranch(4, widths, blocks, use_se)
        self.color = ColorBranch() if use_color else None
        fused = self.texture.out_dim + (self.color.out_dim if use_color else 0)
        self.head = nn.Sequential(
            nn.Linear(fused, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.apply(init_weights)

    def forward(self, x):
        feats = [self.texture(x)]
        if self.color is not None:
            feats.append(self.color(x))
        return self.head(torch.cat(feats, dim=1))   # logits (B, num_classes)