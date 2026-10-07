"""
Tiền xử lý ảnh thịt cá ngừ: tách nền (mask) + chuyển sang không gian màu Lab.

Ý tưởng tách nền:
    - Nền (giấy trắng, thường ám XANH DƯƠNG): SÁNG, kênh a ~ 0, kênh b ÂM.
    - Thịt cá ngừ: ĐỎ (a > 0), ngả vàng/hồng (b >= 0), hoặc TỐI (thịt sẫm, da, vết đen).
    => "điểm thịt" = độ đỏ + độ ngả vàng + độ tối; ngưỡng tự động bằng Otsu.
    (Không dùng "độ đậm màu" chung vì nền ám xanh cũng đậm màu.)
    Sau đó: làm mịn bằng phép toán hình thái, giữ vùng liên thông lớn nhất, lấp lỗ.
"""
import cv2
import numpy as np


def resize_max(img, max_side):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def fill_holes(mask):
    """Lấp các lỗ bên trong vùng thịt (vd: vân mỡ trắng, vùng phản chiếu).
    Tô kín đường viền ngoài -> đúng cả khi miếng thịt chạm mép ảnh."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled


def make_mask(bgr):
    """Trả về mask uint8 (255 = thịt, 0 = nền) cùng kích thước ảnh."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1] - 128, lab[..., 2] - 128
    redness = np.clip(a, 0, None)                   # thịt đỏ: a > 0
    warmth = np.clip(b, 0, None)                    # thịt ngả vàng/hồng; nền xanh có b < 0
    darkness = np.clip(255 - L, 0, None)            # thịt sẫm / da / vết đen: L thấp
    score = 1.5 * redness + 0.8 * warmth + 0.5 * darkness
    score = cv2.GaussianBlur(score, (0, 0), 2)
    score = cv2.normalize(score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = max(3, (min(mask.shape) // 60) | 1)          # kích thước kernel theo cỡ ảnh
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:                                        # giữ vùng lớn nhất
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == biggest, 255, 0).astype(np.uint8)
    return fill_holes(mask)


def white_balance(bgr, mask, target=225, min_frac=0.03):
    """Cân bằng trắng theo NỀN GIẤY TRẮNG.

    Nền đáng lẽ màu trắng trung tính; nếu ảnh bị ám xanh/vàng hay tối/sáng hơn,
    tính hệ số cho từng kênh B, G, R để nền về đúng (target, target, target).
    Cùng hệ số áp lên cả miếng thịt -> màu thịt giữa các buổi chụp so sánh được.

    Trả về (ảnh đã cân bằng, (gB, gG, gR)); nếu nền quá ít -> giữ nguyên, hệ số (1,1,1).
    """
    k = max(3, (min(mask.shape) // 30) | 1)
    near_meat = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = bgr[near_meat == 0].astype(np.float32)              # nền, tránh xa mép thịt
    if len(bg) < min_frac * mask.size:
        return bgr, (1.0, 1.0, 1.0)
    bright = bg.sum(1)
    bg = bg[bright >= np.percentile(bright, 50)]              # bỏ bóng đổ, vết máu trên giấy
    ref = np.median(bg, axis=0).clip(1, 255)                  # màu nền tham chiếu (B, G, R)
    gains = np.clip(target / ref, 0.5, 2.5)
    out = np.clip(bgr.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    return out, tuple(float(g) for g in gains)


def preprocess(bgr, max_side=512, wb=True):
    """Ảnh BGR -> (lab uint8 HxWx3, mask uint8 HxW, tỉ lệ thịt, ảnh BGR đã xử lý, hệ số WB)."""
    bgr = resize_max(bgr, max_side)
    mask = make_mask(bgr)
    gains = (1.0, 1.0, 1.0)
    if wb:
        bgr, gains = white_balance(bgr, mask)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab, mask, float((mask > 0).mean()), bgr, gains


def to_model_input(lab, mask):
    """(lab uint8, mask uint8) -> mảng float32 4xHxW đúng quy ước TunaNet:
    3 kênh Lab chia 255 (0..1) + 1 kênh mask (0/1)."""
    x = lab.astype(np.float32) / 255.0
    m = (mask > 0).astype(np.float32)[..., None]
    return np.concatenate([x, m], axis=2).transpose(2, 0, 1)