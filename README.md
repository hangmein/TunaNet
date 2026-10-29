# TunaNet — Phân loại chất lượng thịt cá ngừ

Website tiếng Việt cho phép chọn hoặc kéo thả ảnh mặt cắt đuôi cá ngừ, xem trước,
gửi ảnh đến server Python và xem cấp dự đoán, xác suất từng cấp, thời gian xử lý
và vùng thịt sau cân bằng trắng/tách nền.

## Chạy website

Mở terminal tại thư mục dự án, dùng môi trường Python đã cài PyTorch của dự án:

```powershell
python -m pip install -r requirements-web.txt
python app.py
```

Truy cập **http://127.0.0.1:8000** bằng trình duyệt. Bấm **Chọn ảnh**, chọn một ảnh
trong hộp thoại, rồi bấm **Phân tích ảnh**. Không cần sửa đường dẫn ảnh trong code.
Dừng server bằng `Ctrl+C`.

Nếu chưa có PyTorch, cài bản phù hợp CPU/CUDA theo https://pytorch.org/get-started/locally/.
Các thư viện phục vụ huấn luyện vẫn ở `requirements.txt`; `requirements-web.txt` chỉ
chứa các thư viện bổ sung cần cho website.

### Trọng số mô hình

**Repository hiện không chứa checkpoint `.pt`.** Website vẫn mở và xem trước ảnh
được; API trả thông báo thiếu mô hình khi gửi ảnh, không tạo kết quả giả.

Mặc định server tìm `outputs/runs/wb_on_rev/best.pt`, giống script dự đoán hiện có.
Để dùng checkpoint khác, chỉ định một lần khi khởi động:

```powershell
python app.py --model "D:\duong-dan\best.pt"
```

Hoặc đặt biến môi trường `TUNANET_MODEL`. Đường dẫn tương đối được tính từ thư mục
dự án. Sau khi thêm/đổi trọng số, khởi động lại server. Chỉ dùng checkpoint đáng tin
cậy do `scripts/train.py` xuất ra: dictionary có `model`, `classes`, `args`.
Nhãn và thứ tự các cấp được đọc từ checkpoint, không tự gán ý nghĩa A/B/C.

Có thể tìm trọng số trong thư mục làm việc cũ, hoặc huấn luyện bằng dữ liệu đã
tiền xử lý và chia tập:

```powershell
python scripts/preprocess.py
python scripts/train.py --epochs 40 --image-size 256 --name tunanet
python app.py --model outputs/runs/tunanet/best.pt
```

Chạy các lệnh trên từ thư mục gốc `TunaNet`. Nếu gặp lỗi thiếu
`data/processed/processed.csv`, chạy `scripts/preprocess.py` trước: script đọc
`data/roboflow_crops/crops.csv` và tạo ảnh Lab, mask cùng CSV đã cân bằng trắng.
Thư mục `data/processed_raw` là bộ dữ liệu riêng; không đổi tên nó để thay thế
cho bộ đã cân bằng trắng dùng với website.

Lệnh train đòi hỏi `data/processed/processed.csv` và `data/splits/split.csv` cùng ảnh
tương ứng. Quy trình chuẩn bị dữ liệu có trong các script bên dưới.

## Cấu trúc và luồng xử lý

- `src/models/blocks.py`: Conv, residual và squeeze-and-excitation.
- `src/models/texture_branch.py`: nhánh CNN kết cấu với MaskedGAP.
- `src/models/color_branch.py`: histogram, thống kê màu Lab trong mask.
- `src/models/tunanet.py`: ghép hai nhánh để phân loại.
- `src/data/preprocess.py`: tìm mask, cân bằng trắng theo nền và chuyển Lab.
- `src/data/dataset.py`: đọc dữ liệu, letterbox, tăng cường hình học.
- `scripts/roboflow_to_crops.py`: cắt miếng thịt từ nhãn YOLO.
- `scripts/analyze_times.py`, `scripts/make_splits.py`: phân tích/chia theo buổi chụp.
- `scripts/preprocess.py`, `scripts/preview_crops.py`: tiền xử lý và xem dữ liệu.
- `scripts/train.py`: huấn luyện, lưu checkpoint và đánh giá.
- `scripts/predict.py`: dự đoán bằng dòng lệnh và các hàm dùng chung với website.
- `app.py`: Flask API, nạp mô hình một lần, chạy Waitress, tự chọn CUDA/CPU.
- `web/templates/index.html`, `web/static/`: giao diện responsive, CSS và JavaScript.
- `tests/`: kiểm tra khối mạng và luồng API website.
- `check_env.py`, `tao_cau_truc.py`, `run_all.ps1`: kiểm tra môi trường, tạo cấu trúc, chạy quy trình.

`POST /api/predict` nhận multipart field `image`, kiểm tra nội dung ảnh, sửa hướng
EXIF rồi gọi `predict_one`: cắt vùng thịt → cân bằng trắng → Lab + mask → letterbox
→ TunaNet → softmax. Trả JSON chứa `prediction`, `confidence`, `probabilities`,
`processed_image`, `meat_ratio`, `box`, `warning`, `elapsed_ms`.
`GET /api/health` báo mô hình đã nạp thành công hay chưa.

Ảnh JPG/PNG/WEBP/BMP tối đa 10 MB, 20 megapixel, mỗi cạnh từ 32 pixel, tỉ lệ cạnh
không quá 20:1. Ảnh và kết quả chỉ xử lý trong bộ nhớ, không ghi ra đĩa. Các lượt
suy luận được khóa tuần tự để tránh chạy đồng thời trên cùng mô hình.

Quy trình web giữ nguyên cân bằng trắng `wb=True` của CLI; dùng checkpoint huấn luyện
trên dữ liệu đã cân bằng trắng để phù hợp. Mô hình luôn đưa ra một cấp kể cả với ảnh
không phải cá ngừ. Xác suất softmax không phải độ chính xác của mô hình trên thực tế.

## Kiểm thử

```powershell
python -m unittest discover -s tests -p test_web.py -v
```

Test tạo checkpoint ngẫu nhiên trong thư mục tạm, chạy HTTP và suy luận thực, so
sánh xác suất với CLI, rồi xóa checkpoint. Điều này chỉ kiểm tra tích hợp, không
đánh giá độ chính xác phân loại. Không dùng trọng số kiểm thử làm mô hình thực tế.

## Chạy trên máy chủ

```powershell
python app.py --host 0.0.0.0 --port 8000 --model outputs/runs/tunanet/best.pt
```

Mặc định chỉ lắng nghe localhost. Với máy chủ/LAN, cần mở cổng tương ứng; với Internet,
cấu hình reverse proxy HTTPS và kiểm soát truy cập phù hợp. Website cần Python/PyTorch
chạy trên server; không thể chỉ tải HTML lên hosting tĩnh để có suy luận.

