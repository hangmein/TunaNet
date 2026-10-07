"""
Chuyển bộ dữ liệu Roboflow (định dạng YOLO, bài toán phát hiện đối tượng)
thành ảnh crop từng miếng thịt, xếp theo cấp chất lượng -> dùng cho TunaNet.

Đầu vào: thư mục giải nén từ file zip Roboflow (định dạng "YOLOv8"):
    tuna_roboflow/
        data.yaml
        train/images/*.jpg   train/labels/*.txt
        valid/images/...     valid/labels/...
        test/images/...      test/labels/...

Đầu ra:
    data/roboflow_crops/Grade-A/xxx.jpg, Grade-B/..., Grade-C/...
    data/roboflow_crops/crops.csv   (file, grade, group_id, split gốc, ảnh nguồn)

group_id: Roboflow đặt tên ảnh tăng cường dạng  <tên_gốc>_jpg.rf.<mã>.jpg
          => các bản sao (lật, xoay...) của cùng một ảnh gốc có chung group_id.
          Khi chia train/val/test PHẢI chia theo group_id để tránh rò rỉ dữ liệu.

Chạy từ thư mục TunaNet:
    python scripts/roboflow_to_crops.py --src D:/Downloads/tuna_roboflow
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import cv2
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def group_id_of(stem):
    """'IMG_0123_jpg.rf.8f3a...' -> 'IMG_0123_jpg'  (bỏ phần mã của Roboflow)."""
    return re.split(r"\.rf\.", stem)[0]


def read_yolo_labels(txt_path):
    """Đọc nhãn YOLO, trả về [(class, cx, cy, w, h)] (tọa độ chuẩn hóa 0..1).

    Hỗ trợ 2 dạng dòng:
      - Khung chữ nhật : class cx cy w h
      - Đa giác        : class x1 y1 x2 y2 ... xn yn  -> lấy khung bao quanh đa giác
    """
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, nums = int(float(parts[0])), list(map(float, parts[1:]))
        if len(nums) == 4:                                   # khung chữ nhật
            boxes.append((c, *nums))
        elif len(nums) >= 6 and len(nums) % 2 == 0:          # đa giác
            xs, ys = nums[0::2], nums[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            boxes.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return boxes


def diagnose(src):
    """In thông tin giúp tìm lỗi khi không cắt được ảnh nào."""
    print("\n[CHẨN ĐOÁN] Không cắt được ảnh nào. Thông tin thư mục:")
    for split in ("train", "valid", "test"):
        d = src / split
        if not d.exists():
            print(f"   {split}/ : KHÔNG có")
            continue
        sub = sorted(p.name for p in d.iterdir() if p.is_dir())
        n_lbl = len(list((d / "labels").glob("*.txt"))) if (d / "labels").exists() else 0
        print(f"   {split}/ : thư mục con = {sub}, số file nhãn .txt = {n_lbl}")
    sample = next(src.rglob("*.txt"), None)
    if sample:
        lines = sample.read_text().splitlines()[:2]
        print(f"   File nhãn mẫu: {sample.relative_to(src)}")
        for ln in lines:
            print(f"      '{ln[:120]}{'...' if len(ln) > 120 else ''}'  ({len(ln.split())} số)")
    else:
        print("   Không tìm thấy file .txt nào -> có thể đã tải sai định dạng (cần YOLOv8).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thư mục giải nén từ zip Roboflow")
    ap.add_argument("--out", default="data/roboflow_crops")
    ap.add_argument("--pad", type=float, default=0.05, help="nới rộng khung thêm 5%% mỗi cạnh")
    ap.add_argument("--min-size", type=int, default=64, help="bỏ crop nhỏ hơn N pixel")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    names = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):                       # một số bản YOLO lưu dạng {0: 'A', ...}
        names = [names[k] for k in sorted(names)]
    print(f"Các cấp: {names}")

    rows, skipped, n_imgs = [], 0, 0
    for split in ("train", "valid", "test"):
        img_dir = src / split / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXT:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            n_imgs += 1
            H, W = img.shape[:2]
            label_path = src / split / "labels" / (img_path.stem + ".txt")
            for k, (c, cx, cy, w, h) in enumerate(read_yolo_labels(label_path)):
                bw, bh = w * W * (1 + 2 * args.pad), h * H * (1 + 2 * args.pad)
                x1, y1 = int(max(cx * W - bw / 2, 0)), int(max(cy * H - bh / 2, 0))
                x2, y2 = int(min(cx * W + bw / 2, W)), int(min(cy * H + bh / 2, H))
                if min(x2 - x1, y2 - y1) < args.min_size:
                    skipped += 1
                    continue
                grade = names[c]
                (out / grade).mkdir(parents=True, exist_ok=True)
                fname = f"{img_path.stem}_{k}.jpg"
                cv2.imwrite(str(out / grade / fname), img[y1:y2, x1:x2],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"file": f"{grade}/{fname}", "grade": grade,
                             "group_id": group_id_of(img_path.stem),
                             "rf_split": split, "source": img_path.name})

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "crops.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "grade", "group_id", "rf_split", "source"])
        writer.writeheader()
        writer.writerows(rows)

    # ---- Báo cáo ----
    if not rows:
        diagnose(src)
        return
    groups = {r["group_id"] for r in rows}
    print(f"\nĐã đọc {n_imgs} ảnh -> {len(rows)} crop (bỏ {skipped} crop quá nhỏ)")
    print(f"Số ảnh GỐC khác nhau (group_id): {len(groups)}")
    print("Số crop mỗi cấp:")
    for g, n in sorted(Counter(r["grade"] for r in rows).items()):
        n_grp = len({r["group_id"] for r in rows if r["grade"] == g})
        print(f"   {g:<10} {n:>5} crop   từ {n_grp:>4} ảnh gốc")

    # Cảnh báo rò rỉ: cùng ảnh gốc xuất hiện ở nhiều split của Roboflow
    splits_of = {}
    for r in rows:
        splits_of.setdefault(r["group_id"], set()).add(r["rf_split"])
    leak = sum(1 for s in splits_of.values() if len(s) > 1)
    if leak:
        print(f"\n[CẢNH BÁO] {leak} ảnh gốc xuất hiện ở NHIỀU split (train/valid/test) của Roboflow."
              "\n           -> Không dùng split gốc của Roboflow, phải tự chia lại theo group_id.")
    print(f"\nĐã lưu vào: {out.resolve()}")


if __name__ == "__main__":
    main()