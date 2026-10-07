"""
Chia dữ liệu train / val / test THEO BUỔI CHỤP (tránh rò rỉ và tránh học "điều kiện buổi chụp").

Phân tích bộ Tuna (Roboflow) cho thấy ảnh được chụp liên tục theo 3 buổi:
    Buổi 1 (15/05 chiều): chỉ cấp A
    Buổi 2 (15/05 chiều): cấp B, C
    Buổi 3 (16/05 trưa) : đủ A, B, C
=> Chia ngẫu nhiên sẽ cho kết quả ảo. Cách trung thực: test trên buổi chụp mô hình chưa thấy.

    test : các buổi trong --test-sessions (mặc định: buổi 3)
    train/val : các buổi còn lại; val = đoạn thời gian CUỐI của mỗi cấp trong mỗi buổi
                (một khối liền nhau, hạn chế ảnh chụp liên tiếp lọt sang val)

Chạy từ thư mục TunaNet:
    python scripts/make_splits.py                                   # test = buổi 3
    python scripts/make_splits.py --test-sessions 1 2 --out data/splits/split_rev.csv   # chiều ngược lại

Kết quả: data/splits/split.csv  (file, grade, group_id, session, split)
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

TS = re.compile(r"(\d{14})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/roboflow_crops/crops.csv")
    ap.add_argument("--out", default="data/splits/split.csv")
    ap.add_argument("--session-gap", type=float, default=600, help="cách nhau > N giây = buổi chụp mới")
    ap.add_argument("--test-sessions", type=int, nargs="+", default=[3])
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.crops, encoding="utf-8")))
    print(f"Đọc {len(rows)} crop từ {args.crops}")

    # --- cấp + thời gian của mỗi ảnh gốc ---
    grades_of = defaultdict(Counter)
    for r in rows:
        grades_of[r["group_id"]][r["grade"]] += 1
    photos = []
    for g, c in grades_of.items():
        m = TS.search(g)
        if not m:
            raise SystemExit(f"Không đọc được thời gian từ tên ảnh: {g}")
        photos.append((datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), g, c.most_common(1)[0][0]))
    photos.sort()

    # --- đánh số buổi chụp ---
    session_of, s = {}, 1
    for i, (t, g, _) in enumerate(photos):
        if i and (t - photos[i - 1][0]).total_seconds() > args.session_gap:
            s += 1
        session_of[g] = s
    n_sessions = s

    # --- gán split ---
    split_of = {}
    blocks = defaultdict(list)                 # (buổi, cấp) -> ảnh theo thứ tự thời gian
    for t, g, grade in photos:
        if session_of[g] in args.test_sessions:
            split_of[g] = "test"
        else:
            blocks[(session_of[g], grade)].append(g)
    for key, gs in blocks.items():
        n_val = round(len(gs) * args.val_frac)
        for i, g in enumerate(gs):
            split_of[g] = "val" if i >= len(gs) - n_val else "train"

    # --- ghi file ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "grade", "group_id", "session", "split"])
        w.writeheader()
        for r in rows:
            g = r["group_id"]
            w.writerow({"file": r["file"], "grade": r["grade"], "group_id": g,
                        "session": session_of[g], "split": split_of[g]})

    # --- báo cáo ---
    print(f"\nSố buổi chụp: {n_sessions}   (test = buổi {args.test_sessions})")
    grades = sorted({p[2] for p in photos})
    print("\nẢnh gốc theo buổi và cấp:")
    for ss in range(1, n_sessions + 1):
        cnt = Counter(p[2] for p in photos if session_of[p[1]] == ss)
        tag = "TEST " if ss in args.test_sessions else "train"
        print(f"   Buổi {ss} [{tag}]  " + "  ".join(f"{gr}={cnt.get(gr, 0):>3}" for gr in grades))

    print("\nSố crop mỗi tập:")
    print("            " + "".join(f"{sp:>9}" for sp in ("train", "val", "test")))
    for gr in grades:
        cells = [sum(1 for r in rows if r["grade"] == gr and split_of[r["group_id"]] == sp)
                 for sp in ("train", "val", "test")]
        print(f"{gr:<12}" + "".join(f"{c:>9}" for c in cells))

    missing = [(sp, gr) for sp in ("train", "val", "test") for gr in grades
               if not any(r["grade"] == gr and split_of[r["group_id"]] == sp for r in rows)]
    for sp, gr in missing:
        print(f"[!] Tập {sp} không có ảnh {gr}")
    leak = len({r["group_id"] for r in rows if split_of[r["group_id"]] == "test"}
               & {r["group_id"] for r in rows if split_of[r["group_id"]] != "test"})
    print(f"\n[{'OK' if leak == 0 else 'LỖI'}] Không ảnh gốc nào vừa ở test vừa ở train/val")
    print(f"Đã lưu: {out.resolve()}")


if __name__ == "__main__":
    main()