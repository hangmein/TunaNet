"""
Kiểm tra nhánh kết cấu + đo tốc độ trên máy.
Chạy từ thư mục TunaNet:   python tests/test_texture_branch.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.models.blocks import init_weights
from src.models.texture_branch import TextureBranch, MaskedGAP

torch.manual_seed(0)
torch.set_num_threads(8)
print("=" * 80)


def fake_batch(B, size):
    """Ảnh giả: 3 kênh Lab ngẫu nhiên + mask hình elip ở giữa (giả lập mặt cắt thịt)."""
    lab = torch.randn(B, 3, size, size)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    ellipse = ((xx / 0.8) ** 2 + (yy / 0.55) ** 2 <= 1).float()
    mask = ellipse.expand(B, 1, size, size).clone()
    return torch.cat([lab, mask], dim=1)


# 1. MaskedGAP bỏ qua hoàn toàn vùng nền
gap = MaskedGAP()
feat = torch.randn(2, 8, 16, 16)
mask = torch.zeros(2, 1, 16, 16)
mask[..., :, :8] = 1                  # nửa trái là thịt
out1 = gap(feat, mask)
feat2 = feat.clone()
feat2[..., :, 8:] += 1000             # sửa mạnh nửa phải (nền)
out2 = gap(feat2, mask)
ok = torch.allclose(out1, out2)
print(f"[{'OK' if ok else 'LỖI'}] MaskedGAP: thay đổi vùng nền KHÔNG ảnh hưởng kết quả")
assert ok

# 2. Kích thước đầu ra của cả nhánh
net = TextureBranch()
net.apply(init_weights)
n_params = sum(p.numel() for p in net.parameters())
for size in (128, 256, 384):
    x = fake_batch(2, size)
    y = net(x)
    fm = net.feature_map(x)
    ok = tuple(y.shape) == (2, 256)
    print(f"[{'OK' if ok else 'LỖI'}] Ảnh {size}x{size}: feature map {tuple(fm.shape)} -> vector {tuple(y.shape)}")
    assert ok
print(f"      Tổng tham số nhánh kết cấu: {n_params:,}")

# 3. Đo tốc độ train (forward + backward) trên máy này
print("-" * 80)
device = "cuda" if torch.cuda.is_available() else "cpu"
net = net.to(device).train()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
for size, B in ((128, 16), (256, 8)):
    x = fake_batch(B, size).to(device)
    for i in range(4):                 # lần đầu chậm (khởi động), đo 3 lần sau
        if i == 1:
            t0 = time.perf_counter()
        opt.zero_grad()
        net(x).pow(2).mean().backward()
        opt.step()
    dt = (time.perf_counter() - t0) / 3 / B
    print(f"[ĐO]  {device.upper()} ảnh {size}x{size}: {dt*1000:.0f} ms/ảnh khi train"
          f"  -> ~{dt*1000/60:.1f} phút / epoch 1.000 ảnh")

print("=" * 80)
print("Nhánh kết cấu hoạt động đúng!")