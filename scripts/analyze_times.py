"""
Phân tích thời gian chụp của bộ dữ liệu (tên ảnh dạng IMG20220516121300).

Mục đích: hiểu ảnh được chụp theo buổi như thế nào, để chọn cách chia dữ liệu hợp lý.

Chạy từ thư mục TunaNet:   python scripts/analyze_times.py
"""
import csv
import re
from collections import Counter, defaultdict
from datetime import datetime

TS = re.compile(r"(\d{14})")
SESSION_GAP = 10 * 60   # cách nhau > 10 phút = buổi chụp mới

rows = list(csv.DictReader(open("data/roboflow_crops/crops.csv", encoding="utf-8")))
grade_of = {}
for r in rows:
    grade_of.setdefault(r["group_id"], Counter())[r["grade"]] += 1
photos = []
for g, c in grade_of.items():
    m = TS.search(g)
    if m:
        photos.append((datetime.strptime(m.group(1), "%Y%m%d%H%M%S"), c.most_common(1)[0][0], g))
photos.sort()
print(f"{len(photos)} ảnh gốc có thời gian, từ {photos[0][0]} đến {photos[-1][0]}\n")

# 1. Khoảng cách giữa 2 ảnh liên tiếp (tất cả các cấp, theo thứ tự thời gian)
gaps = [(b[0] - a[0]).total_seconds() for a, b in zip(photos, photos[1:])]
bins = [0, 5, 10, 15, 20, 30, 45, 60, 120, 600, 1e12]
labels = ["0-5s", "5-10s", "10-15s", "15-20s", "20-30s", "30-45s", "45-60s", "1-2ph", "2-10ph", ">10ph"]
hist = Counter()
for x in gaps:
    for lo, hi, lb in zip(bins, bins[1:], labels):
        if lo <= x < hi:
            hist[lb] += 1
print("Khoảng cách giữa 2 ảnh liên tiếp:")
for lb in labels:
    print(f"   {lb:>7}: {hist[lb]:>4}  {'#' * (hist[lb] // 4)}")

# 2. Ảnh liên tiếp có đổi cấp không? (đổi cấp liên tục = chụp xen kẽ; ít đổi = chụp theo lô)
changes = sum(1 for a, b in zip(photos, photos[1:]) if a[1] != b[1])
print(f"\nSố lần đổi cấp giữa 2 ảnh liên tiếp: {changes} / {len(photos) - 1}")

# 3. Các buổi chụp
print(f"\nCác buổi chụp (cách nhau > {SESSION_GAP // 60} phút):")
sessions, cur = [], [photos[0]]
for a, b in zip(photos, photos[1:]):
    if (b[0] - a[0]).total_seconds() > SESSION_GAP:
        sessions.append(cur)
        cur = []
    cur.append(b)
sessions.append(cur)
for i, s in enumerate(sessions, 1):
    cnt = Counter(p[1] for p in s)
    print(f"   Buổi {i}: {s[0][0]:%d/%m %H:%M} -> {s[-1][0]:%H:%M}  {len(s):>4} ảnh   "
          + "  ".join(f"{g}={cnt.get(g, 0)}" for g in sorted({p[1] for p in photos})))

# 4. Đoạn chụp liên tục cùng cấp (run)
runs, cur = [], [photos[0]]
for a, b in zip(photos, photos[1:]):
    if b[1] == a[1]:
        cur.append(b)
    else:
        runs.append(cur)
        cur = [b]
runs.append(cur)
lens = sorted(len(r) for r in runs)
print(f"\nSố đoạn chụp liên tiếp cùng cấp: {len(runs)}  (dài nhất {lens[-1]} ảnh, trung vị {lens[len(lens) // 2]} ảnh)")