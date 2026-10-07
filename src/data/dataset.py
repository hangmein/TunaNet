"""
Dataset cho TunaNet: đọc ảnh Lab + mask đã tiền xử lý, ghép với file chia tập.

Mỗi mẫu trả về:
    x : tensor float32 (4, S, S) = Lab/255 + mask   (đúng quy ước TunaNet)
    y : nhãn số (0, 1, 2 ...) theo thứ tự tên cấp
Ảnh được thu về cạnh dài = S, giữ tỉ lệ, phần thiếu đệm 0 (mask = 0 nên không ảnh hưởng MaskedGAP).
Augmentation CHỈ hình học (lật, xoay, phóng to nhỏ) - KHÔNG đổi màu vì màu là tín hiệu phân loại.
"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_split(processed_csv, split_csv, split):
    """Trả về danh sách (đường_dẫn_lab, đường_dẫn_mask, cấp) của một tập."""
    proc_dir = Path(processed_csv).parent
    proc = {r["file"]: r for r in csv.DictReader(open(processed_csv, encoding="utf-8"))}
    items = []
    for r in csv.DictReader(open(split_csv, encoding="utf-8")):
        if r["split"] == split and r["file"] in proc:
            p = proc[r["file"]]
            items.append((proc_dir / p["lab"], proc_dir / p["mask"], r["grade"]))
    return items


def letterbox(img, size, interp):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = img
    return out


def augment(lab, mask):
    """Biến đổi hình học giống nhau cho ảnh và mask."""
    if random.random() < 0.5:
        lab, mask = lab[:, ::-1], mask[:, ::-1]
    if random.random() < 0.5:
        lab, mask = lab[::-1], mask[::-1]
    k = random.randint(0, 3)
    lab, mask = np.rot90(lab, k), np.rot90(mask, k)
    lab, mask = np.ascontiguousarray(lab), np.ascontiguousarray(mask)

    h, w = mask.shape
    angle = random.uniform(-15, 15)
    scale = random.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    lab = cv2.warpAffine(lab, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return lab, mask


class TunaDataset(Dataset):
    def __init__(self, items, classes, image_size=256, train=False):
        self.items = items
        self.classes = list(classes)
        self.size = image_size
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        lab_path, mask_path, grade = self.items[i]
        lab = cv2.imread(str(lab_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        lab = letterbox(lab, self.size, cv2.INTER_AREA)
        mask = letterbox(mask, self.size, cv2.INTER_NEAREST)
        if self.train:
            lab, mask = augment(lab, mask)
        x = np.concatenate([lab.astype(np.float32) / 255.0,
                            (mask > 127).astype(np.float32)[..., None]], axis=2)
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        return x, self.classes.index(grade)