"""
DỰ ĐOÁN TRỌN LUỒNG:  Ảnh -> YOLO-seg (tìm thịt + mask) -> OpenCV (cắt, cân bằng trắng, Lab) -> TunaNet -> A/B/C

Chạy từ thư mục TunaNet:
    python scripts/predict_full.py --input test_image
    python scripts/predict_full.py --input test_image\\anh1.jpg
    python scripts/predict_full.py --input test_image --model outputs/runs/yolo_s1/best.pt

Kết quả: outputs/predictions_full/<tên ảnh>_pred.jpg  +  predictions.csv
Mỗi bước xử lý giống HỆT lúc tạo dữ liệu train (scripts/preprocess_yolo.py).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from preprocess_yolo import crop_box, yolo_mask  # noqa: E402  (dùng chung code với lúc train)
from src.data.preprocess import make_mask, resize_max, white_balance  # noqa: E402
from src.models.tunanet import TunaNet  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = [(60, 180, 75), (0, 165, 255), (60, 60, 230)]   # A xanh lá, B cam, C đỏ (BGR)


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def imwrite(p, img):
    cv2.imencode(Path(p).suffix, img)[1].tofile(str(p))


def letterbox(img, size, interp):
    """Giống src/data/dataset.py: thu cạnh dài về size, giữ tỉ lệ, đệm 0."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = img
    return out


def load_tunanet(path, device):
    ck = torch.load(path, map_location=device)
    a = ck["args"]
    model = TunaNet(num_classes=len(ck["classes"]), use_color=not a.get("no_color", False),
                    use_se=not a.get("no_se", False))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["classes"], a.get("image_size", 256)


@torch.no_grad()
def run_pipeline(bgr, yolo, tunanet, image_size, device, conf=0.25, margin=0.15, tight=0.05, max_side=512):
    t = {}
    # ---- Bước 1: YOLO-seg ----
    t0 = time.perf_counter()
    mask_full, yconf = yolo_mask(yolo, bgr, conf)
    src = "yolo"
    if mask_full is None:                       # YOLO không thấy -> dự phòng bằng ngưỡng màu
        mask_full, src = make_mask(bgr), "opencv_fallback"
    t["yolo"] = time.perf_counter() - t0

    # ---- Bước 2: OpenCV (giống preprocess_yolo.py) ----
    t0 = time.perf_counter()
    x1, y1, x2, y2 = crop_box(mask_full, margin)
    wide, mwide = bgr[y1:y2, x1:x2], mask_full[y1:y2, x1:x2]
    wide, _ = white_balance(wide, mwide)
    tx1, ty1, tx2, ty2 = crop_box(mwide, tight)
    crop, mcrop = wide[ty1:ty2, tx1:tx2], mwide[ty1:ty2, tx1:tx2]
    crop = resize_max(crop, max_side)
    mcrop = cv2.resize(mcrop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab_l = letterbox(lab, image_size, cv2.INTER_AREA)
    mask_l = letterbox(mcrop, image_size, cv2.INTER_NEAREST)
    x = np.concatenate([lab_l.astype(np.float32) / 255.0,
                        (mask_l > 127).astype(np.float32)[..., None]], axis=2).transpose(2, 0, 1)
    t["opencv"] = time.perf_counter() - t0

    # ---- Bước 3: TunaNet ----
    t0 = time.perf_counter()
    prob = torch.softmax(tunanet(torch.from_numpy(x)[None].to(device)), 1)[0].cpu().numpy()
    t["tunanet"] = time.perf_counter() - t0

    return {"prob": prob, "mask_full": mask_full, "box": (x1, y1, x2, y2), "crop": crop, "mcrop": mcrop,
            "mask_src": src, "yolo_conf": yconf, "time": t}


def draw(bgr, res, classes):
    prob = res["prob"]
    k = int(prob.argmax())
    color = COLORS[k % len(COLORS)]
    H = 480
    # Trái: ảnh gốc + mask YOLO
    left = bgr.copy()
    m = res["mask_full"] > 0
    left[m] = (0.65 * left[m] + 0.35 * np.array(color)).astype(np.uint8)
    cnts, _ = cv2.findContours(res["mask_full"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left, cnts, -1, color, max(2, bgr.shape[1] // 250))
    left = cv2.resize(left, (int(left.shape[1] * H / left.shape[0]), H))
    # Phải: vùng thịt sau OpenCV (cân bằng trắng) + viền mask
    right = res["crop"].copy()
    cnts, _ = cv2.findContours(res["mcrop"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(right, cnts, -1, (0, 255, 255), 2)
    right = cv2.resize(right, (int(right.shape[1] * H / right.shape[0]), H))
    canvas = np.hstack([left, np.full((H, 10, 3), 255, np.uint8), right])
    # Dưới: kết quả + xác suất
    bar = np.full((80 + 32 * len(classes), canvas.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"Ket qua: {classes[k]}  ({prob[k] * 100:.1f}%)", (12, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    for i, (c, p) in enumerate(zip(classes, prob)):
        y = 70 + 32 * i
        cv2.putText(bar, c, (12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.rectangle(bar, (130, y + 4), (130 + int(400 * p), y + 24), COLORS[i % len(COLORS)], -1)
        cv2.putText(bar, f"{p * 100:.1f}%", (540, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    tt = res["time"]
    cv2.putText(bar, f"YOLO {tt['yolo'] * 1000:.0f}ms | OpenCV {tt['opencv'] * 1000:.0f}ms | "
                     f"TunaNet {tt['tunanet'] * 1000:.0f}ms", (640, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1)
    return np.vstack([canvas, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="1 ảnh hoặc 1 thư mục ảnh")
    ap.add_argument("--yolo", default="runs/segment/outputs/yolo/seg_meat1/weights/best.pt")
    ap.add_argument("--model", default="outputs/runs/yolo_s2/best.pt", help="trọng số TunaNet")
    ap.add_argument("--out", default="outputs/predictions_full")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    for p, name in ((args.yolo, "YOLO-seg"), (args.model, "TunaNet")):
        if not Path(p).exists():
            raise SystemExit(f"Không thấy mô hình {name}: {p}")
    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"KHÔNG TÌM THẤY: '{src}' (đang đứng ở {Path('.').resolve()})")
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    if not files:
        raise SystemExit(f"Thư mục '{src}' không có ảnh nào")

    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(args.yolo)
    tunanet, classes, image_size = load_tunanet(args.model, device)
    # Làm nóng: lần chạy đầu tiên luôn chậm (khởi tạo), không tính vào thời gian đo
    dummy = np.full((480, 640, 3), 220, np.uint8)
    cv2.ellipse(dummy, (320, 240), (150, 100), 0, 0, 360, (40, 40, 160), -1)
    run_pipeline(dummy, yolo, tunanet, image_size, device, conf=args.conf)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"YOLO-seg: {args.yolo}\nTunaNet : {args.model} | Cấp: {classes}\n{len(files)} ảnh\n")

    rows = []
    for f in files:
        bgr = imread(f)
        if bgr is None:
            print(f"   [bỏ qua] không đọc được {f.name}")
            continue
        res = run_pipeline(bgr, yolo, tunanet, image_size, device, conf=args.conf)
        prob, k = res["prob"], int(res["prob"].argmax())
        total = sum(res["time"].values()) * 1000
        note = "  [!] YOLO không thấy thịt, dùng ngưỡng màu" if res["mask_src"] != "yolo" else ""
        print(f"{f.name:<38} -> {classes[k]:<8} {prob[k] * 100:5.1f}%   "
              + "  ".join(f"{c}={p * 100:.0f}%" for c, p in zip(classes, prob)) + f"   ({total:.0f} ms){note}")
        imwrite(out / f"{f.stem}_pred.jpg", draw(bgr, res, classes))
        rows.append({"file": f.name, "pred": classes[k], "confidence": f"{prob[k]:.4f}",
                     **{f"p_{c}": f"{p:.4f}" for c, p in zip(classes, prob)},
                     "mask_src": res["mask_src"], "yolo_conf": f"{res['yolo_conf']:.3f}",
                     **{f"ms_{s}": f"{v * 1000:.1f}" for s, v in res["time"].items()}})

    if rows:
        with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"\nĐã lưu ảnh kết quả vào: {out.resolve()}")


if __name__ == "__main__":
    main()