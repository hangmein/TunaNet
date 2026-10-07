"""
Kiểm tra môi trường cho đề tài phân loại chất lượng thịt cá ngừ.
Chạy:  python check_env.py
"""
import importlib
import sys

OK, FAIL = "[OK]  ", "[LỖI]"


def check_import(name, attr="__version__"):
    try:
        mod = importlib.import_module(name)
        print(f"{OK} {name:<16} {getattr(mod, attr, '')}")
        return mod
    except Exception as e:
        print(f"{FAIL} {name:<16} chưa cài được: {e}")
        return None


print("=" * 60)
print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
print("=" * 60)

torch = check_import("torch")
check_import("torchvision")
np = check_import("numpy")
cv2 = check_import("cv2")
for lib in ["pandas", "sklearn", "PIL", "albumentations",
            "matplotlib", "tqdm", "torchinfo", "pytorch_grad_cam"]:
    check_import(lib)

# ---- Kiểm tra GPU ----
print("-" * 60)
device = "cpu"
if torch is not None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"{OK} GPU NVIDIA: {torch.cuda.get_device_name(0)}")
        print(f"      CUDA (bản PyTorch dùng): {torch.version.cuda}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"      VRAM: {mem:.1f} GB")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        print(f"{OK} GPU Apple Silicon (MPS)")
    else:
        print("[!]   Không thấy GPU -> sẽ train bằng CPU (chậm). "
              "Có thể dùng Google Colab/Kaggle để train.")

# ---- Thử một lớp tích chập (giống TunaNet: 4 kênh Lab + mask) ----
if torch is not None:
    try:
        x = torch.randn(2, 4, 128, 128, device=device)
        conv = torch.nn.Conv2d(4, 32, 3, padding=1).to(device)
        y = conv(x)
        print(f"{OK} Chạy thử Conv2d trên {device}: {tuple(x.shape)} -> {tuple(y.shape)}")
    except Exception as e:
        print(f"{FAIL} Chạy thử Conv2d thất bại: {e}")

# ---- Thử chuyển đổi màu sang Lab bằng OpenCV ----
if cv2 is not None and np is not None:
    try:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:] = (40, 40, 180)  # BGR: màu đỏ giống thịt cá
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        print(f"{OK} OpenCV BGR->Lab: pixel đỏ -> Lab = {lab[0, 0].tolist()}")
    except Exception as e:
        print(f"{FAIL} Chuyển màu Lab thất bại: {e}")

print("=" * 60)
print("Xong. Nếu mọi dòng đều [OK] là môi trường đã sẵn sàng.") 