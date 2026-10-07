"""
Kiểm tra nhánh màu + TunaNet hoàn chỉnh.
Chạy từ thư mục TunaNet:   python tests/test_tunanet.py

Bài kiểm tra quan trọng nhất là số 4 (OVERFIT): cho mạng học thuộc một bộ dữ liệu giả nhỏ.
Nếu mạng KHÔNG học thuộc được => code có lỗi, chưa được train thật.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from src.models.color_branch import masked_color_features
from src.models.tunanet import TunaNet

torch.manual_seed(0)
torch.set_num_threads(8)
print("=" * 80)


def fake_batch(labels, size=64):
    """Ảnh Lab giả trong [0,1] + mask elip. Kênh a (độ đỏ) phụ thuộc vào cấp chất lượng:
    cấp 0 = đỏ tươi, cấp 1 = đỏ vừa, cấp 2 = nhợt/nâu."""
    B = len(labels)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    mask = ((xx / 0.8) ** 2 + (yy / 0.55) ** 2 <= 1).float().expand(B, 1, size, size).clone()
    lab = 0.5 + 0.05 * torch.randn(B, 3, size, size)
    a_level = torch.tensor([0.75, 0.62, 0.55])[labels]          # độ đỏ theo cấp
    lab[:, 1] += (a_level - 0.5).view(B, 1, 1)
    lab = lab.clamp(0, 1)
    lab = lab * mask + (1 - mask) * torch.rand(B, 3, size, size)  # nền ngẫu nhiên
    return torch.cat([lab, mask], dim=1)


# 1. Đặc trưng màu bỏ qua vùng nền
x = fake_batch(torch.tensor([0, 1]))
f1 = masked_color_features(x)
x2 = x.clone()
bg = x2[:, 3:4] < 0.5
x2[:, :3] = torch.where(bg.expand(-1, 3, -1, -1), torch.rand_like(x2[:, :3]), x2[:, :3])
f2 = masked_color_features(x2)
ok = torch.allclose(f1, f2) and f1.shape == (2, 63)
print(f"[{'OK' if ok else 'LỖI'}] Nhánh màu: {tuple(f1.shape)} đặc trưng, thay nền KHÔNG đổi kết quả")
assert ok
print(f"      Kênh a trung bình: cấp 0 = {f1[0, 49]:.3f}, cấp 1 = {f1[1, 49]:.3f} (cấp 0 phải đỏ hơn)")

# 2. Kích thước đầu ra + số tham số của các biến thể (dùng cho bảng ablation)
for name, kw in [("TunaNet đầy đủ", {}),
                 ("Không nhánh màu", {"use_color": False}),
                 ("Không SE", {"use_se": False})]:
    m = TunaNet(num_classes=3, **kw).eval()
    y = m(fake_batch(torch.tensor([0, 1])))
    n = sum(p.numel() for p in m.parameters())
    ok = tuple(y.shape) == (2, 3)
    print(f"[{'OK' if ok else 'LỖI'}] {name:<18} đầu ra {tuple(y.shape)}   {n:,} tham số")
    assert ok

# 3. Cấu trúc mạng (torchinfo)
try:
    from torchinfo import summary
    print("-" * 80)
    summary(TunaNet(3), input_size=(2, 4, 256, 256), depth=2,
            col_names=("output_size", "num_params"))
except ImportError:
    print("(Bỏ qua in cấu trúc: chưa cài torchinfo)")

# 4. OVERFIT: mạng phải học thuộc 24 ảnh giả
print("-" * 80)
labels = torch.arange(3).repeat(8)          # 24 ảnh, 8 ảnh mỗi cấp
data = fake_batch(labels)
model = TunaNet(num_classes=3).train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
t0 = time.perf_counter()
for step in range(1, 61):
    opt.zero_grad()
    loss = loss_fn(model(data), labels)
    loss.backward()
    opt.step()
    if step % 15 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(data).argmax(1) == labels).float().mean().item()
        model.train()
        print(f"      bước {step:>2}: loss = {loss.item():.4f}   độ chính xác = {acc*100:.0f}%")
ok = acc == 1.0
print(f"[{'OK' if ok else 'LỖI'}] Overfit: mạng học thuộc dữ liệu nhỏ ({time.perf_counter()-t0:.0f} giây)")
assert ok, "Mạng không học được -> kiểm tra lại code"

print("=" * 80)
print("TunaNet hoàn chỉnh hoạt động đúng!")