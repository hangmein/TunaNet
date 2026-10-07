"""
Dự đoán cấp chất lượng cho ẢNH MỚI bằng mô hình TunaNet đã train.

Chạy từ thư mục TunaNet:
    python scripts/predict.py --input duong_dan_anh.jpg
    python scripts/predict.py --input thu_muc_anh/
    python scripts/predict.py --input thu_muc_anh/ --model outputs/runs/wb_on/best.pt

Quy trình cho mỗi ảnh (giống hệt lúc train):
    1. Tìm miếng thịt (mask) trên toàn ảnh -> cắt quanh miếng thịt (thay cho bước YOLO)
    2. Cân bằng trắng theo nền giấy -> chuyển Lab -> mask
    3. Đưa vào TunaNet -> xác suất từng cấp

Kết quả: outputs/predictions/<tên ảnh>_pred.jpg  +  predictions.csv
Lưu ý: ảnh nên chụp giống bộ dữ liệu (miếng thịt cắt ngang đuôi, trên nền sáng).
       Mô hình LUÔN trả về một cấp, kể cả với ảnh không phải thịt cá ngừ.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import letterbox  # noqa: E402
from src.data.preprocess import make_mask, preprocess, resize_max  # noqa: E402
from src.models.tunanet import TunaNet  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {0: (60, 180, 75), 1: (0, 165, 255), 2: (60, 60, 230)}   # A xanh lá, B cam, C đỏ (BGR)


def crop_to_meat(bgr, margin=0.15):
    """Tìm miếng thịt trên ảnh lớn rồi cắt quanh nó (chừa lề để còn nền cho cân bằng trắng)."""
    small = resize_max(bgr, 800)
    m = make_mask(small)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return bgr, None
    s = bgr.shape[1] / small.shape[1]
    x1, x2, y1, y2 = xs.min() * s, xs.max() * s, ys.min() * s, ys.max() * s
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    H, W = bgr.shape[:2]
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
    return bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    a = ck["args"]
    model = TunaNet(num_classes=len(ck["classes"]), use_color=not a.get("no_color", False),
                    use_se=not a.get("no_se", False))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["classes"], a.get("image_size", 256)


@torch.no_grad()
def predict_one(model, bgr, image_size, device):
    crop, box = crop_to_meat(bgr)
    lab, mask, ratio, bgr_wb, _ = preprocess(crop, 512, wb=True)
    lab_l = letterbox(lab, image_size, cv2.INTER_AREA)
    mask_l = letterbox(mask, image_size, cv2.INTER_NEAREST)
    x = np.concatenate([lab_l.astype(np.float32) / 255.0,
                        (mask_l > 127).astype(np.float32)[..., None]], axis=2).transpose(2, 0, 1)
    prob = torch.softmax(model(torch.from_numpy(x)[None].to(device)), 1)[0].cpu().numpy()
    return prob, box, bgr_wb, mask, ratio


def draw(bgr, box, prob, classes, wb_crop, mask):
    k = int(prob.argmax())
    color = COLORS.get(k, (255, 255, 255))
    vis = resize_max(bgr.copy(), 900)
    s = vis.shape[1] / bgr.shape[1]
    if box:
        x1, y1, x2, y2 = [int(v * s) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
    # Ảnh phụ: miếng thịt sau cân bằng trắng + viền mask
    side = wb_crop.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(side, cnts, -1, (0, 255, 255), 2)
    side = cv2.resize(side, (int(side.shape[1] * vis.shape[0] / side.shape[0]), vis.shape[0]))
    canvas = np.hstack([vis, side])
    bar = np.full((70 + 32 * len(classes), canvas.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, f"Ket qua: {classes[k]}  ({prob[k] * 100:.1f}%)", (12, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    for i, (c, p) in enumerate(zip(classes, prob)):
        y = 75 + 32 * i
        cv2.putText(bar, f"{c:<8}", (12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.rectangle(bar, (130, y + 4), (130 + int(400 * p), y + 24), COLORS.get(i, (128, 128, 128)), -1)
        cv2.putText(bar, f"{p * 100:.1f}%", (540, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return np.vstack([canvas, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="1 ảnh hoặc 1 thư mục ảnh")
    ap.add_argument("--model", default="outputs/runs/wb_on_rev/best.pt")
    ap.add_argument("--out", default="outputs/predictions")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, image_size = load_model(args.model, device)
    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT) if src.is_dir() else [src]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Mô hình: {args.model} | Cấp: {classes} | {len(files)} ảnh\n")

    rows = []
    for f in files:
        bgr = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)   # đọc được đường dẫn tiếng Việt
        if bgr is None:
            print(f"   [bỏ qua] không đọc được {f.name}")
            continue
        prob, box, wb_crop, mask, ratio = predict_one(model, bgr, image_size, device)
        k = int(prob.argmax())
        note = "  [!] mask bất thường, kiểm tra ảnh" if ratio < 0.15 or ratio > 0.97 else ""
        print(f"{f.name:<40} -> {classes[k]:<8} {prob[k] * 100:5.1f}%   "
              + "  ".join(f"{c}={p * 100:.0f}%" for c, p in zip(classes, prob)) + note)
        vis = draw(bgr, box, prob, classes, wb_crop, mask)
        cv2.imencode(".jpg", vis)[1].tofile(str(out / f"{f.stem}_pred.jpg"))
        rows.append({"file": f.name, "pred": classes[k], "confidence": f"{prob[k]:.4f}",
                     **{f"p_{c}": f"{p:.4f}" for c, p in zip(classes, prob)}})

    if rows:
        with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    print(f"\nĐã lưu ảnh kết quả vào: {out.resolve()}")


if __name__ == "__main__":
    main()