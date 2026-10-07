"""
Nhánh kết cấu (Texture Branch) của TunaNet.

    Đầu vào : (B, 4, H, W)  = 3 kênh Lab + 1 kênh mask (1 = thịt, 0 = nền)
    Đầu ra  : (B, 256)      = vector đặc trưng kết cấu của vùng thịt

    Stem   : ConvBNReLU 4->32, stride 2           H/2
    Tầng 1 : 2 x ResidualBlock 32                 H/2
    Tầng 2 : 2 x ResidualBlock 64,  khối đầu s=2  H/4
    Tầng 3 : 2 x ResidualBlock 128, khối đầu s=2  H/8
    Tầng 4 : 2 x ResidualBlock 256, khối đầu s=2  H/16
    MaskedGAP: trung bình mỗi kênh, CHỈ trên pixel thịt  -> 256
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBNReLU, ResidualBlock


class MaskedGAP(nn.Module):
    """Global Average Pooling chỉ trên vùng mask.

    GAP thường lấy trung bình cả ảnh -> nền trắng cũng bị tính vào.
    MaskedGAP thu nhỏ mask về đúng kích thước feature map, rồi:
        out[c] = sum(feat[c] * mask) / sum(mask)
    """

    def forward(self, feat, mask):
        # mask (B,1,H,W) -> (B,1,h,w); mode="area" cho ra tỉ lệ thịt trong mỗi ô (0..1)
        m = F.interpolate(mask, size=feat.shape[-2:], mode="area")
        num = (feat * m).sum(dim=(2, 3))            # (B, C)
        den = m.sum(dim=(2, 3)).clamp_min(1e-6)     # (B, 1), tránh chia cho 0
        return num / den


def make_stage(in_ch, out_ch, n_blocks, stride, use_se=True):
    """Một tầng = n_blocks ResidualBlock; chỉ khối đầu tiên đổi kênh / giảm kích thước."""
    layers = [ResidualBlock(in_ch, out_ch, stride, use_se)]
    for _ in range(n_blocks - 1):
        layers.append(ResidualBlock(out_ch, out_ch, 1, use_se))
    return nn.Sequential(*layers)


class TextureBranch(nn.Module):
    def __init__(self, in_ch=4, widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2), use_se=True):
        super().__init__()
        self.stem = ConvBNReLU(in_ch, widths[0], 3, stride=2)
        stages, prev = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            stride = 1 if i == 0 else 2
            stages.append(make_stage(prev, w, n, stride, use_se))
            prev = w
        self.stages = nn.Sequential(*stages)
        self.pool = MaskedGAP()
        self.out_dim = widths[-1]

    def forward(self, x):
        mask = x[:, 3:4]           # kênh thứ 4 là mask
        feat = self.stages(self.stem(x))
        return self.pool(feat, mask)

    def feature_map(self, x):
        """Trả về feature map trước pooling (dùng cho Grad-CAM, xem mạng nhìn vào đâu)."""
        return self.stages(self.stem(x))