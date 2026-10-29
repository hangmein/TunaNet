"""
LUỒNG MỚI - Bước 1+2:  YOLO-seg tìm miếng thịt + mask  ->  OpenCV xử lý vùng thịt.

Cho mỗi ảnh GỐC trong bộ Roboflow:
    1. YOLO-seg  : tìm miếng thịt, lấy mask theo đường biên (đối tượng tin cậy nhất)
    2. OpenCV    : cắt quanh miếng thịt (chừa lề có nền giấy) -> resize -> cân bằng trắng
                   theo nền -> chuyển Lab -> lưu ảnh Lab + mask
    3. Đo chất lượng mask: so mask YOLO và mask ngưỡng màu cũ với NHÃN THẬT (IoU)

Đầu ra có cùng định dạng với scripts/preprocess.py -> train.py dùng lại được ngay:
    python scripts/train.py --processed data/processed_yolo/processed.csv --name yolo_wb_on

Chạy từ thư mục TunaNet:
    python scripts/preprocess_yolo.py --src D:\\StudyUni\\AI\\tuna_roboflow
"""
import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import make_mask, resize_max, white_balance  # noqa: E402

DEFAULT_MODEL = "runs/segment/outputs/yolo/seg_meat1/weights/best.pt"


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, img):
    cv2.imencode(Path(p).suffix, img)[1].tofile(str(p))


def gt_mask(label_path, h, w):
    """Mask từ nhãn đa giác thật của Roboflow (để đo IoU)."""
    m = np.zeros((h, w), np.uint8)
    if not label_path.exists():
        return None
    for line in label_path.read_text().splitlines():
        nums = list(map(float, line.split()[1:]))
        if len(nums) >= 6 and len(nums) % 2 == 0:
            pts = np.array([[x * w, y * h] for x, y in zip(nums[0::2], nums[1::2])], np.int32)
            cv2.fillPoly(m, [pts], 255)
        elif len(nums) == 4:
            cx, cy, bw, bh = nums
            cv2.rectangle(m, (int((cx - bw / 2) * w), int((cy - bh / 2) * h)),
                          (int((cx + bw / 2) * w), int((cy + bh / 2) * h)), 255, -1)
    return m if m.any() else None


def iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def yolo_mask(model, bgr, conf):
    """Mask (cùng kích thước ảnh gốc) của miếng thịt tin cậy nhất; None nếu không thấy."""
    r = model.predict(bgr, conf=conf, verbose=False, retina_masks=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return None, 0.0
    i = int(r.boxes.conf.argmax())
    poly = r.masks.xy[i]
    if len(poly) < 3:
        return None, 0.0
    m = np.zeros(bgr.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    return m, float(r.boxes.conf[i])


def crop_box(mask, margin):
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx + 1), min(H, y2 + my + 1)


def overlay(bgr, mask):
    vis = bgr.copy()
    m = mask > 0
    vis[m] = (0.6 * vis[m] + np.array([0, 100, 0])).clip(0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 2)
    return vis


def tile(img, size=220):
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    t = np.full((size, size, 3), 128, np.uint8)
    t[(size - img.shape[0]) // 2:(size - img.shape[0]) // 2 + img.shape[0],
      (size - img.shape[1]) // 2:(size - img.shape[1]) // 2 + img.shape[1]] = img
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thư mục giải nén Roboflow")
    ap.add_argument("--crops", default="data/roboflow_crops/crops.csv",
                    help="để giữ đúng tên file/cấp như dữ liệu TunaNet cũ (khớp split.csv)")
    ap.add_argument("--split", default="data/splits/split.csv", help="chỉ để in IoU theo từng tập")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="data/processed_yolo")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.15, help="lề RỘNG khi cân bằng trắng (cần nền giấy)")
    ap.add_argument("--tight", type=float, default=0.05, help="lề SÁT khi cắt ảnh đưa vào TunaNet (giống khung Roboflow)")
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--no-wb", action="store_true")
    ap.add_argument("--check-n", type=int, default=4)
    args = ap.parse_args()

    from ultralytics import YOLO
    if not Path(args.model).exists():
        raise SystemExit(f"Không thấy mô hình YOLO: {args.model}\n-> dùng --model <đường dẫn tới best.pt>")
    model = YOLO(args.model)

    src, out = Path(args.src), Path(args.out)
    rows = list(csv.DictReader(open(args.crops, encoding="utf-8")))
    split_of = {}
    if Path(args.split).exists():
        split_of = {r["group_id"]: r["split"] for r in csv.DictReader(open(args.split, encoding="utf-8"))}

    random.seed(0)
    check = {g: set(random.sample([r["file"] for r in rows if r["grade"] == g],
                                  min(args.check_n, sum(r["grade"] == g for r in rows))))
             for g in sorted({r["grade"] for r in rows})}
    check_tiles = defaultdict(list)

    results, ious = [], defaultdict(lambda: {"yolo": [], "cv": []})
    n_fallback = n_skip = 0
    for r in tqdm(rows, desc="YOLO-seg + OpenCV"):
        k = int(re.search(r"_(\d+)\.jpg$", r["file"]).group(1))
        if k > 0:                      # ảnh có nhiều khung: luồng mới chỉ lấy miếng tin cậy nhất
            n_skip += 1
            continue
        img_path = src / r["rf_split"] / "images" / r["source"]
        bgr = imread(img_path)
        if bgr is None:
            n_skip += 1
            continue

        # ---- Bước 1: YOLO-seg ----
        mask_full, conf = yolo_mask(model, bgr, args.conf)
        mask_src = "yolo"
        if mask_full is None:          # dự phòng: không thấy thì dùng ngưỡng màu cũ
            mask_full, mask_src = make_mask(bgr), "opencv_fallback"
            n_fallback += 1

        # ---- Đo IoU với nhãn thật (so YOLO và ngưỡng màu cũ) ----
        gt = gt_mask(src / r["rf_split"] / "labels" / (Path(r["source"]).stem + ".txt"), *bgr.shape[:2])
        if gt is not None:
            sp = split_of.get(r["group_id"], "?")
            ious[sp]["yolo"].append(iou(mask_full, gt))
            ious[sp]["cv"].append(iou(make_mask(bgr), gt))

        # ---- Bước 2: OpenCV ----
        # (a) Cắt RỘNG (chừa nhiều nền giấy) để cân bằng trắng có đủ nền tham chiếu
        x1, y1, x2, y2 = crop_box(mask_full, args.margin)
        wide, mwide = bgr[y1:y2, x1:x2], mask_full[y1:y2, x1:x2]
        gains = (1.0, 1.0, 1.0)
        if not args.no_wb:
            wide, gains = white_balance(wide, mwide)
        # (b) Cắt SÁT lại quanh miếng thịt -> thịt chiếm phần lớn ảnh, giữ chi tiết thớ thịt
        tx1, ty1, tx2, ty2 = crop_box(mwide, args.tight)
        crop, mcrop = wide[ty1:ty2, tx1:tx2], mwide[ty1:ty2, tx1:tx2]
        crop = resize_max(crop, args.max_side)
        mcrop = cv2.resize(mcrop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        stem = Path(r["file"]).with_suffix("")
        (out / stem.parent).mkdir(parents=True, exist_ok=True)
        lab_path, mask_path = f"{stem}.png", f"{stem}_mask.png"
        imwrite(out / lab_path, lab)
        imwrite(out / mask_path, mcrop)
        results.append({"file": r["file"], "grade": r["grade"], "lab": lab_path, "mask": mask_path,
                        "meat_ratio": f"{(mcrop > 0).mean():.3f}",
                        "wb_gains": " ".join(f"{g:.3f}" for g in gains),
                        "mask_src": mask_src, "yolo_conf": f"{conf:.3f}"})
        if r["file"] in check[r["grade"]]:
            check_tiles[r["grade"]].append(np.hstack([tile(overlay(bgr, mask_full)), tile(overlay(crop, mcrop))]))

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "processed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # Ảnh kiểm tra: [ảnh gốc + mask YOLO | vùng đã cắt + cân bằng trắng + mask]
    lines = []
    for g, tiles in check_tiles.items():
        label = np.full((220, 130, 3), 255, np.uint8)
        cv2.putText(label, g, (8, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        tiles += [np.full((220, 440, 3), 255, np.uint8)] * (args.check_n - len(tiles))
        lines.append(np.hstack([label] + tiles))
    fig = Path("outputs/figures") / f"{out.name}_check.jpg"
    fig.parent.mkdir(parents=True, exist_ok=True)
    imwrite(fig, np.vstack(lines))

    report = [f"Đã xử lý {len(results)} ảnh -> {out.resolve()}",
              f"Mask từ YOLO: {len(results) - n_fallback}   |   phải dùng ngưỡng màu dự phòng: {n_fallback}"
              + (f"   |   bỏ qua: {n_skip}" if n_skip else ""),
              f"Lề cân bằng trắng: {args.margin:.0%}   |   lề cắt đưa vào TunaNet: {args.tight:.0%}",
              "",
              "Độ khớp mask với NHÃN THẬT (IoU trung bình, 1.0 = khớp hoàn toàn):",
              f"   {'tập':<8}{'số ảnh':>8}{'YOLO-seg':>11}{'Ngưỡng màu':>13}"]
    for sp in ("train", "val", "test", "?"):
        if ious[sp]["yolo"]:
            y, c = np.array(ious[sp]["yolo"]), np.array(ious[sp]["cv"])
            report.append(f"   {sp:<8}{len(y):>8}{y.mean():>11.3f}{c.mean():>13.3f}"
                          f"     (IoU < 0.8: YOLO {int((y < 0.8).sum())} ảnh, ngưỡng màu {int((c < 0.8).sum())} ảnh)")
    report.append(f"\nẢnh kiểm tra: {fig.resolve()}")
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"(Bảng trên đã được lưu vào {(out / 'report.txt').resolve()})")


if __name__ == "__main__":
    main()