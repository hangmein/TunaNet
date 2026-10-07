"""
Tiền xử lý toàn bộ ảnh crop: tạo mask + CÂN BẰNG TRẮNG theo nền giấy + chuyển Lab.
Đồng thời xuất ảnh kiểm tra để xem bằng mắt.

Chạy từ thư mục TunaNet:
    python scripts/preprocess.py                                        # có cân bằng trắng
    python scripts/preprocess.py --no-wb --out data/processed_raw       # không cân bằng (để so sánh)

Kết quả:
    data/processed/<cấp>/<tên>.png        ảnh Lab (lưu thô 3 kênh, KHÔNG xem bằng trình xem ảnh)
    data/processed/<cấp>/<tên>_mask.png   mask (trắng = thịt)
    data/processed/processed.csv          file, grade, lab, mask, meat_ratio
    outputs/figures/<tên>_check.jpg       ảnh gốc | sau cân bằng trắng + mask, vài ảnh mỗi cấp
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import preprocess  # noqa: E402


def overlay(bgr, mask):
    vis = bgr.copy()
    green = np.zeros_like(bgr)
    green[..., 1] = 255
    m = mask > 0
    vis[m] = (0.6 * vis[m] + 0.4 * green[m]).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
    return vis


def tile(img, size):
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    t = np.full((size, size, 3), 128, np.uint8)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    t[y:y + img.shape[0], x:x + img.shape[1]] = img
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/roboflow_crops")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--check-n", type=int, default=4, help="số ảnh mỗi cấp trong ảnh kiểm tra")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wb", action="store_true", help="tắt cân bằng trắng")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    rows = list(csv.DictReader(open(src / "crops.csv", encoding="utf-8")))
    random.seed(args.seed)
    check = {}
    for g in sorted({r["grade"] for r in rows}):
        files = [r["file"] for r in rows if r["grade"] == g]
        check[g] = set(random.sample(files, min(args.check_n, len(files))))

    results, check_rows, bad = [], {g: [] for g in check}, []
    for r in tqdm(rows, desc="Tiền xử lý"):
        bgr = cv2.imread(str(src / r["file"]))
        if bgr is None:
            continue
        lab, mask, ratio, bgr_wb, gains = preprocess(bgr, args.max_side, wb=not args.no_wb)
        stem = Path(r["file"]).with_suffix("")
        (out / stem.parent).mkdir(parents=True, exist_ok=True)
        lab_path, mask_path = f"{stem}.png", f"{stem}_mask.png"
        cv2.imwrite(str(out / lab_path), lab)
        cv2.imwrite(str(out / mask_path), mask)
        results.append({"file": r["file"], "grade": r["grade"], "lab": lab_path,
                        "mask": mask_path, "meat_ratio": f"{ratio:.3f}",
                        "wb_gains": " ".join(f"{g:.3f}" for g in gains)})
        if ratio < 0.15 or ratio > 0.97:
            bad.append((r["file"], ratio))
        if r["file"] in check[r["grade"]]:
            small = cv2.resize(bgr, (lab.shape[1], lab.shape[0])) if bgr.shape[:2] != lab.shape[:2] else bgr
            check_rows[r["grade"]].append(np.hstack([tile(small, 220), tile(overlay(bgr_wb, mask), 220)]))

    with open(out / "processed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "grade", "lab", "mask", "meat_ratio", "wb_gains"])
        w.writeheader()
        w.writerows(results)

    # Ảnh kiểm tra: mỗi hàng 1 cấp, mỗi ô = [gốc | mask]
    lines = []
    for g, tiles in check_rows.items():
        label = np.full((220, 130, 3), 255, np.uint8)
        cv2.putText(label, g, (8, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        tiles += [np.full((220, 440, 3), 255, np.uint8)] * (args.check_n - len(tiles))
        lines.append(np.hstack([label] + tiles))
    fig = Path("outputs/figures") / f"{out.name}_check.jpg"
    fig.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(fig), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 88])

    ratios = np.array([float(r["meat_ratio"]) for r in results])
    print(f"\nĐã xử lý {len(results)} ảnh -> {out.resolve()}")
    print(f"Tỉ lệ diện tích thịt trong ảnh: trung vị {np.median(ratios):.2f}, "
          f"nhỏ nhất {ratios.min():.2f}, lớn nhất {ratios.max():.2f}")
    print(f"Ảnh nghi tách nền lỗi (thịt < 15% hoặc > 97% ảnh): {len(bad)}")
    for f_, r_ in bad[:10]:
        print(f"   {f_}  ({r_:.2f})")
    if not args.no_wb:
        g = np.array([[float(v) for v in r["wb_gains"].split()] for r in results])
        print(f"Hệ số cân bằng trắng (B, G, R) trung bình: {g.mean(0).round(2)}  "
              f"-> ảnh gốc ám {'xanh' if g[:, 0].mean() < g[:, 2].mean() else 'vàng/đỏ'}")
    print(f"Ảnh kiểm tra: {fig.resolve()}")


if __name__ == "__main__":
    main()