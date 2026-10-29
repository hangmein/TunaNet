"""
Chuẩn bị dữ liệu cho YOLO (Ultralytics) từ bộ Roboflow, dùng ĐÚNG cách chia theo buổi chụp
của TunaNet (đọc từ data/splits/split.csv) -> so sánh công bằng.

--task seg    : YOLO PHÂN VÙNG - giữ nguyên đa giác viền miếng thịt (luồng chính:
                YOLO-seg tìm thịt + mask -> OpenCV -> TunaNet)
--task detect : YOLO PHÁT HIỆN - đổi đa giác thành khung chữ nhật (dùng để so sánh)

--classes 1   : 1 lớp "meat"   -> YOLO chỉ TÌM miếng thịt, việc phân loại để TunaNet làm
--classes 3   : giữ 3 lớp A/B/C -> YOLO làm cả phân loại (đối chứng)

Chạy từ thư mục TunaNet:
    python scripts/prepare_yolo.py --src D:\\StudyUni\\AI\\tuna_roboflow --task seg --classes 1 --preview 6
    (chiều ngược: thêm --split data/splits/split_rev.csv)

Kết quả: data/yolo/<task>_<cls>_<split>/images/{train,val,test}, labels/..., data.yaml
         outputs/figures/<tên>_labels.jpg (nếu dùng --preview): ảnh có vẽ viền nhãn để kiểm tra
"""
import argparse
import csv
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def group_id_of(stem):
    """'IMG_0123_jpg.rf.8f3a...' -> 'IMG_0123_jpg' (giống roboflow_to_crops.py)."""
    return re.split(r"\.rf\.", stem)[0]


def parse_line(line):
    """Đọc 1 dòng nhãn YOLO -> (lớp, danh sách điểm [(x,y),...]) chuẩn hóa 0..1.
    Khung chữ nhật (cx cy w h) được đổi thành đa giác 4 góc."""
    parts = line.split()
    if len(parts) < 5:
        return None
    c, nums = int(float(parts[0])), list(map(float, parts[1:]))
    if len(nums) == 4:
        cx, cy, w, h = nums
        pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
               (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
    elif len(nums) >= 6 and len(nums) % 2 == 0:
        pts = list(zip(nums[0::2], nums[1::2]))
    else:
        return None
    pts = [(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)) for x, y in pts]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
        return None
    return c, pts


def to_text(cls_id, pts, task):
    if task == "seg":
        return f"{cls_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return f"{cls_id} {cx:.6f} {cy:.6f} {max(xs) - min(xs):.6f} {max(ys) - min(ys):.6f}"


def place(src, dst):
    """Tạo hard link cho đỡ tốn ổ đĩa; không được thì copy."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def draw_preview(samples, task, path, size=320):
    """Vẽ viền/khung nhãn lên vài ảnh để kiểm tra bằng mắt."""
    tiles = []
    for img_path, objs in samples:
        img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        for _, pts in objs:
            p = np.array([[x * w, y * h] for x, y in pts], np.int32)
            if task == "seg":
                over = img.copy()
                cv2.fillPoly(over, [p], (0, 200, 0))
                img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
                cv2.polylines(img, [p], True, (0, 255, 255), max(2, w // 200))
            else:
                x, y, bw, bh = cv2.boundingRect(p)
                cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 255), max(2, w // 200))
        s = size / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        tile = np.full((size, size, 3), 128, np.uint8)
        tile[:img.shape[0], :img.shape[1]] = img
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.full((size, size, 3), 255, np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", sheet)[1].tofile(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thư mục giải nén Roboflow (có data.yaml)")
    ap.add_argument("--split", default="data/splits/split.csv")
    ap.add_argument("--task", choices=["seg", "detect"], default="seg")
    ap.add_argument("--classes", type=int, choices=[1, 3], default=1)
    ap.add_argument("--out", default=None, help="mặc định: data/yolo/<task>_<cls>_<tên split>")
    ap.add_argument("--preview", type=int, default=0, help="vẽ viền nhãn lên N ảnh ngẫu nhiên để kiểm tra")
    args = ap.parse_args()

    src = Path(args.src)
    names = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    one = args.classes == 1
    tag = f"{args.task}_{'meat1' if one else 'grade3'}_{Path(args.split).stem}"
    out = Path(args.out or f"data/yolo/{tag}").resolve()

    split_of = {r["group_id"]: r["split"] for r in csv.DictReader(open(args.split, encoding="utf-8"))}
    if not split_of:
        raise SystemExit(f"File chia rỗng: {args.split}. Hãy chạy make_splits.py trước.")
    for sp in ("train", "val", "test"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    n_img, n_obj, n_pts = Counter(), Counter(), []
    cls_count = {sp: Counter() for sp in ("train", "val", "test")}
    skipped_nolabel = skipped_nosplit = 0
    samples = []
    for rf_split in ("train", "valid", "test"):
        img_dir = src / rf_split / "images"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            sp = split_of.get(group_id_of(img.stem))
            if sp is None:
                skipped_nosplit += 1
                continue
            lbl = src / rf_split / "labels" / (img.stem + ".txt")
            objs = []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    p = parse_line(line)
                    if p:
                        objs.append(p)
            if not objs:
                skipped_nolabel += 1
                continue
            place(img, out / "images" / sp / img.name)
            text = [to_text(0 if one else c, pts, args.task) for c, pts in objs]
            (out / "labels" / sp / (img.stem + ".txt")).write_text("\n".join(text) + "\n")
            for c, pts in objs:
                cls_count[sp][names[c]] += 1
                n_pts.append(len(pts))
            n_img[sp] += 1
            n_obj[sp] += len(objs)
            samples.append((out / "images" / sp / img.name, objs))

    data_yaml = {"path": str(out), "train": "images/train", "val": "images/val", "test": "images/test",
                 "names": {0: "meat"} if one else {i: n for i, n in enumerate(names)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False),
                                   encoding="utf-8")

    print(f"Nhiệm vụ : {'PHÂN VÙNG (seg) - giữ đa giác viền' if args.task == 'seg' else 'PHÁT HIỆN (detect) - khung chữ nhật'}")
    print(f"Số lớp   : {'1 (meat) - YOLO chỉ tìm miếng thịt' if one else '3 (A/B/C) - YOLO tìm + phân loại'}")
    print(f"Cách chia: {args.split}\n")
    print(f"{'':8}{'ảnh':>7}{'vật thể':>9}   theo cấp gốc")
    for sp in ("train", "val", "test"):
        cc = "  ".join(f"{k}={v}" for k, v in sorted(cls_count[sp].items()))
        print(f"{sp:8}{n_img[sp]:>7}{n_obj[sp]:>9}   {cc}")
    if n_pts:
        print(f"\nSố điểm trên mỗi đa giác: trung vị {int(np.median(n_pts))}, ít nhất {min(n_pts)}, nhiều nhất {max(n_pts)}")
    if skipped_nosplit:
        print(f"[!] Bỏ qua {skipped_nosplit} ảnh không có trong file chia")
    if skipped_nolabel:
        print(f"[!] Bỏ qua {skipped_nolabel} ảnh không có nhãn hợp lệ")
    if args.preview and samples:
        random.seed(0)
        fig = Path("outputs/figures") / f"{tag}_labels.jpg"
        draw_preview(random.sample(samples, min(args.preview, len(samples))), args.task, fig)
        print(f"\nẢnh kiểm tra nhãn: {fig.resolve()}")
    print(f"\nĐã tạo: {out}\nFile cấu hình cho YOLO: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()