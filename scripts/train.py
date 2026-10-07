"""
Huấn luyện + đánh giá TunaNet.

Chạy thử nhanh (kiểm tra quy trình, ~vài phút):
    python scripts/train.py --epochs 3 --image-size 128 --name smoke_test
Chạy thật:
    python scripts/train.py --epochs 40 --image-size 256 --name tunanet_s12_test3
Chiều kiểm chứng ngược:
    python scripts/train.py --split data/splits/split_rev.csv --name tunanet_s3_test12
Ablation (ví dụ bỏ nhánh màu):
    python scripts/train.py --no-color --name ablation_no_color

Kết quả trong outputs/runs/<name>/:
    best.pt          trọng số tốt nhất (theo F1 trên tập val)
    history.csv      loss / độ chính xác từng epoch
    results.json     kết quả trên tập TEST
    confusion.png    ma trận nhầm lẫn trên tập test
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import TunaDataset, load_split  # noqa: E402
from src.models.tunanet import TunaNet  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def save_confusion(cm, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed/processed.csv")
    ap.add_argument("--split", default="data/splits/split.csv")
    ap.add_argument("--name", default="tunanet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=10, help="dừng sớm nếu val F1 không tăng sau N epoch")
    ap.add_argument("--no-color", action="store_true", help="ablation: bỏ nhánh màu")
    ap.add_argument("--no-se", action="store_true", help="ablation: bỏ khối SE")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("outputs/runs") / args.name
    out.mkdir(parents=True, exist_ok=True)

    # ---------- Dữ liệu ----------
    tr = load_split(args.processed, args.split, "train")
    va = load_split(args.processed, args.split, "val")
    te = load_split(args.processed, args.split, "test")
    classes = sorted({g for _, _, g in tr + va + te})
    print(f"Thiết bị: {device} | Cấp: {classes}")
    print(f"Số ảnh  train={len(tr)}  val={len(va)}  test={len(te)}")
    if not tr or not va or not te:
        raise SystemExit("Thiếu dữ liệu ở một tập. Kiểm tra lại preprocess.py và make_splits.py")

    mk = lambda items, train: DataLoader(  # noqa: E731
        TunaDataset(items, classes, args.image_size, train=train),
        batch_size=args.batch, shuffle=train, num_workers=args.workers, drop_last=train and len(items) > args.batch)
    dl_tr, dl_va, dl_te = mk(tr, True), mk(va, False), mk(te, False)

    # ---------- Mô hình ----------
    model = TunaNet(num_classes=len(classes), use_color=not args.no_color, use_se=not args.no_se).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TunaNet: {n_params:,} tham số | màu={not args.no_color} SE={not args.no_se}")

    counts = np.bincount([classes.index(g) for _, _, g in tr], minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Learning rate: tăng dần trong 5% đầu (warmup), sau đó giảm theo hình cos về ~0
    total = max(1, args.epochs * len(dl_tr))
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))

    # ---------- Huấn luyện ----------
    best_f1, best_ep, history = -1, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot_loss, n = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(y)
            n += len(y)
        yv, pv = predict(model, dl_va, device)
        va_acc, va_f1 = accuracy_score(yv, pv), f1_score(yv, pv, average="macro")
        history.append({"epoch": ep, "train_loss": tot_loss / n, "val_acc": va_acc, "val_f1": va_f1})
        flag = ""
        if va_f1 > best_f1:
            best_f1, best_ep, flag = va_f1, ep, "  *lưu"
            torch.save({"model": model.state_dict(), "classes": classes, "args": vars(args)}, out / "best.pt")
        print(f"Epoch {ep:>3}/{args.epochs}  loss={tot_loss / n:.4f}  val_acc={va_acc:.3f}  "
              f"val_F1={va_f1:.3f}  ({time.time() - t0:.0f}s){flag}")
        if ep - best_ep >= args.patience:
            print(f"Dừng sớm: val F1 không tăng sau {args.patience} epoch")
            break

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    # ---------- Đánh giá trên TEST ----------
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    yt, pt = predict(model, dl_te, device)
    cm = confusion_matrix(yt, pt, labels=range(len(classes)))
    res = {"name": args.name, "split": args.split, "best_epoch": best_ep, "val_f1": best_f1,
           "test_acc": accuracy_score(yt, pt), "test_f1_macro": f1_score(yt, pt, average="macro"),
           "test_f1_per_class": dict(zip(classes, f1_score(yt, pt, average=None, labels=range(len(classes))).tolist())),
           "confusion": cm.tolist(), "classes": classes, "params": n_params}
    json.dump(res, open(out / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    save_confusion(cm, classes, out / "confusion.png",
                   f"{args.name}\nTest acc={res['test_acc']:.3f}  F1={res['test_f1_macro']:.3f}")

    print("\n================ KẾT QUẢ TRÊN TẬP TEST ================")
    print(f"Epoch tốt nhất (theo val): {best_ep}")
    print(f"Accuracy = {res['test_acc']:.3f}   Macro-F1 = {res['test_f1_macro']:.3f}")
    for c, v in res["test_f1_per_class"].items():
        print(f"   F1 {c:<10} = {v:.3f}")
    print("Ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán):")
    print("            " + "".join(f"{c:>10}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:<12}" + "".join(f"{v:>10}" for v in row))
    print(f"\nĐã lưu kết quả: {out.resolve()}")


if __name__ == "__main__":
    main()