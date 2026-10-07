"""
Ghép ngẫu nhiên vài ảnh crop của mỗi cấp thành 1 tấm ảnh tổng hợp để xem nhanh.

Chạy từ thư mục TunaNet:
    python scripts/preview_crops.py
    python scripts/preview_crops.py --n 4 --seed 7     # 4 ảnh/cấp, bộ ngẫu nhiên khác

Kết quả: outputs/figures/preview_crops.jpg
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def fit_tile(img, size):
    """Thu ảnh vào ô vuông size x size, giữ tỉ lệ, viền xám."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    tile = np.full((size, size, 3), 128, np.uint8)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    tile[y:y + img.shape[0], x:x + img.shape[1]] = img
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/roboflow_crops")
    ap.add_argument("--n", type=int, default=3, help="số ảnh mỗi cấp")
    ap.add_argument("--size", type=int, default=300, help="kích thước mỗi ô (pixel)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/figures/preview_crops.jpg")
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.src)
    grades = sorted(d for d in src.iterdir() if d.is_dir())
    if not grades:
        raise SystemExit(f"Không thấy thư mục cấp nào trong {src}. Đã chạy roboflow_to_crops.py chưa?")

    label_w, rows = 160, []
    for g in grades:
        files = [f for f in g.iterdir() if f.suffix.lower() in IMG_EXT]
        picks = random.sample(files, min(args.n, len(files)))
        tiles = [fit_tile(cv2.imread(str(f)), args.size) for f in picks]
        tiles += [np.full((args.size, args.size, 3), 255, np.uint8)] * (args.n - len(tiles))
        label = np.full((args.size, label_w, 3), 255, np.uint8)
        cv2.putText(label, g.name, (10, args.size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(label, f"{len(files)} anh", (10, args.size // 2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1)
        rows.append(np.hstack([label] + tiles))
        print(f"{g.name:<10} {len(files):>5} ảnh   mẫu: {', '.join(p.name for p in picks)}")

    sheet = np.vstack(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\nĐã lưu ảnh tổng hợp: {out.resolve()}")


if __name__ == "__main__":
    main()