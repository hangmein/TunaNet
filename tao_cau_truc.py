"""
Tạo cấu trúc thư mục cho dự án TunaNet.
Chạy MỘT LẦN trong thư mục TunaNet:   python tao_cau_truc.py
Script không ghi đè file đã có.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FOLDERS = [
    "data/raw",          # ảnh gốc chụp từ buồng chụp (KHÔNG sửa, KHÔNG xóa)
    "data/processed",    # ảnh đã tiền xử lý: Lab + mask, đã crop
    "data/splits",       # file chia train/val/test (theo con cá)
    "src/models",        # kiến trúc mạng TunaNet
    "src/data",          # đọc dữ liệu, tiền xử lý, augmentation
    "src/utils",         # hàm tiện ích: seed, chọn thiết bị, đo chỉ số
    "scripts",           # các lệnh chạy: tiền xử lý, train, đánh giá
    "tests",             # test từng khối của mạng
    "notebooks",         # Jupyter để thử nghiệm, xem ảnh
    "configs",           # file cấu hình thí nghiệm
    "outputs/checkpoints",  # trọng số mô hình đã train (.pt)
    "outputs/logs",         # log TensorBoard
    "outputs/figures",      # hình: confusion matrix, Grad-CAM...
]

FILES = {
    "src/__init__.py": "",
    "src/models/__init__.py": "",
    "src/data/__init__.py": "",
    "src/utils/__init__.py": "",
    "tests/__init__.py": "",
    "data/labels.csv": "fish_id,image_file,grade,species,weight_kg,state,date,grader\n",
    "configs/base.yaml": (
        "# Cấu hình mặc định cho thí nghiệm\n"
        "seed: 42\n"
        "num_classes: 3          # sửa theo số cấp chất lượng của nhóm\n"
        "image_size: 256         # thử nghiệm 256, train cuối 384\n"
        "batch_size: 16\n"
        "epochs: 50\n"
        "lr: 0.001\n"
        "weight_decay: 0.0001\n"
        "num_threads: 8          # số luồng CPU khi chạy trên laptop\n"
    ),
    ".gitignore": (
        ".venv/\n__pycache__/\n*.pyc\n.ipynb_checkpoints/\n"
        "data/raw/\ndata/processed/\noutputs/\n"
    ),
    "README.md": (
        "# TunaNet - Phân loại chất lượng thịt cá ngừ từ ảnh mặt cắt đuôi\n\n"
        "## Cài đặt\n"
        "1. `python -m venv .venv` rồi kích hoạt môi trường\n"
        "2. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu "
        "--extra-index-url https://pypi.org/simple`\n"
        "3. `pip install -r requirements.txt`\n"
        "4. `python check_env.py`\n\n"
        "## Cấu trúc\n"
        "- `data/raw` ảnh gốc, `data/processed` ảnh đã xử lý, `data/labels.csv` nhãn\n"
        "- `src/` mã nguồn chính, `scripts/` lệnh chạy, `tests/` kiểm tra\n"
        "- `outputs/` kết quả train (không đưa lên git)\n"
    ),
}

for f in FOLDERS:
    (ROOT / f).mkdir(parents=True, exist_ok=True)
    print(f"[thư mục] {f}")

for path, content in FILES.items():
    p = ROOT / path
    if p.exists():
        print(f"[bỏ qua]  {path} (đã có)")
        continue
    p.write_text(content, encoding="utf-8")
    print(f"[tạo]     {path}")

print("\nXong! Cấu trúc dự án TunaNet đã sẵn sàng.")