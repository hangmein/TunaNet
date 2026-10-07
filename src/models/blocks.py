"""
Các khối xây dựng cơ bản của TunaNet.

    ConvBNReLU     : Conv -> BatchNorm -> ReLU  (viên gạch nhỏ nhất)
    SEBlock        : Squeeze-and-Excitation, mạng tự chấm điểm tầm quan trọng của từng kênh
    ResidualBlock  : 2 lớp tích chập + đường tắt (skip connection) + SE

Quy ước kích thước tensor: (B, C, H, W)
    B = số ảnh trong 1 batch, C = số kênh, H x W = chiều cao x chiều rộng
"""
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Conv k x k -> BatchNorm -> ReLU.

    stride=1: giữ nguyên H, W.   stride=2: H, W giảm một nửa.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2  # giữ kích thước khi stride=1
        # bias=False vì BatchNorm ngay sau đã có tham số dịch chuyển (beta)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation.

    1. Squeeze   : lấy trung bình mỗi kênh trên toàn ảnh   -> (B, C, 1, 1)
    2. Excitation: FC thu nhỏ C -> C/r -> ReLU -> FC về C -> Sigmoid  => trọng số 0..1
    3. Nhân trọng số vào từng kênh: kênh quan trọng giữ lại, kênh ít quan trọng bị giảm.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),  # Conv 1x1 trên tensor 1x1 = lớp FC
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))  # (B, C, 1, 1)
        return x * w               # broadcast: nhân trọng số vào mọi pixel của kênh


class ResidualBlock(nn.Module):
    """Khối dư:

        x ─> ConvBNReLU ─> Conv ─> BN ─> [SE] ─(+)─> ReLU ─> ra
        └──────────────── đường tắt ──────────────┘

    Nếu đổi số kênh hoặc stride=2, đường tắt dùng Conv 1x1 để khớp kích thước.
    """

    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, 3, stride)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()  # giữ nguyên x

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + self.shortcut(x)  # cộng đường tắt
        return self.act(out)


def init_weights(module):
    """Khởi tạo He (Kaiming) cho Conv, chuẩn cho BN/Linear.
    Rất quan trọng khi train từ đầu (không có trọng số pretrained).
    Dùng: model.apply(init_weights)
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        nn.init.zeros_(module.bias)