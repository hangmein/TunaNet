"""
Nhánh màu (Color Branch) của TunaNet.

Không dùng tích chập. Tính trực tiếp các thống kê màu Lab TRONG vùng mask:
    - Histogram L, a, b : 3 kênh x 16 bin          = 48 số
    - mean, std, phân vị 10/50/90 của mỗi kênh      = 15 số
    => 63 số -> BatchNorm -> FC 64 -> ReLU          => 64 số

QUY ƯỚC ĐẦU VÀO (bắt buộc giống bước tiền xử lý):
    x[:, 0:3] = Lab của OpenCV (uint8, 0..255) chia cho 255  => giá trị trong [0, 1]
    x[:, 3]   = mask (1 = thịt, 0 = nền)
"""
import torch
import torch.nn as nn

QUANTILES = (0.10, 0.50, 0.90)


def masked_color_features(x, n_bins=16):
    """Trả về (B, 3*n_bins + 3*5) đặc trưng màu, chỉ tính trên pixel thịt."""
    B = x.shape[0]
    lab = x[:, :3].flatten(2)            # (B, 3, HW)
    mask = (x[:, 3:4].flatten(2) > 0.5).float()  # (B, 1, HW)
    n_pix = mask.sum(dim=2).clamp_min(1.0)       # (B, 1)

    # --- Histogram có trọng số mask ---
    idx = (lab.clamp(0, 1) * n_bins).long().clamp(0, n_bins - 1)   # bin của từng pixel
    hist = torch.zeros(B, 3, n_bins, device=x.device, dtype=x.dtype)
    hist.scatter_add_(2, idx, mask.expand_as(lab).to(x.dtype))
    hist = hist / n_pix.unsqueeze(-1)    # chuẩn hóa: tổng mỗi histogram = 1
    hist = hist.flatten(1)               # (B, 48)

    # --- mean, std ---
    mean = (lab * mask).sum(2) / n_pix                               # (B, 3)
    var = (((lab - mean.unsqueeze(-1)) ** 2) * mask).sum(2) / n_pix
    std = var.clamp_min(0).sqrt()                                    # (B, 3)

    # --- phân vị: tính riêng từng ảnh (mỗi ảnh có số pixel thịt khác nhau) ---
    q = torch.tensor(QUANTILES, device=x.device, dtype=x.dtype)
    qs = []
    for i in range(B):
        sel = lab[i][:, mask[i, 0] > 0]                              # (3, số pixel thịt)
        if sel.shape[1] == 0:
            qs.append(torch.zeros(3, len(QUANTILES), device=x.device, dtype=x.dtype))
        else:
            qs.append(torch.quantile(sel, q, dim=1).T)               # (3, 3)
    qs = torch.stack(qs).flatten(1)                                  # (B, 9)

    return torch.cat([hist, mean, std, qs], dim=1)                   # (B, 63)


class ColorBranch(nn.Module):
    def __init__(self, n_bins=16, out_dim=64):
        super().__init__()
        self.n_bins = n_bins
        in_dim = 3 * n_bins + 3 * (2 + len(QUANTILES))
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(in_dim),   # đưa các đặc trưng về cùng thang đo
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        with torch.no_grad():         # thống kê màu không cần gradient
            feats = masked_color_features(x, self.n_bins)
        return self.mlp(feats)