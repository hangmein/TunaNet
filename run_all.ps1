# Chạy toàn bộ quy trình TunaNet từ A -> Z (chế độ chạy thử nhanh).
# Cách chạy (trong VS Code terminal, đã bật .venv, đứng ở thư mục TunaNet):
#     powershell -ExecutionPolicy Bypass -File run_all.ps1
# Mặc định dùng thư mục Roboflow ở D:\StudyUni\AI\tuna_roboflow

param(
    [string]$Roboflow = "D:\StudyUni\AI\tuna_roboflow",
    [int]$Epochs = 3,
    [int]$ImageSize = 128
)
$ErrorActionPreference = "Stop"

# Hiển thị tiếng Việt đúng + luôn chạy trong thư mục chứa script (thư mục TunaNet)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
Set-Location -Path $PSScriptRoot
if (-not (Test-Path "check_env.py")) {
    Write-Host "Không thấy check_env.py cạnh run_all.ps1. Hãy đặt run_all.ps1 trong thư mục TunaNet." -ForegroundColor Red
    exit 1
}

function Step($n, $title, $cmd) {
    Write-Host ""
    Write-Host "==================== BƯỚC $n : $title ====================" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "LỖI ở bước $n ($title). Dừng lại." -ForegroundColor Red
        exit 1
    }
}

Step 1 "Kiểm tra môi trường"        "python check_env.py"
Step 2 "Kiểm tra mô hình TunaNet"   "python tests/test_tunanet.py"
Step 3 "Cắt ảnh từ Roboflow"        "python scripts/roboflow_to_crops.py --src `"$Roboflow`""
Step 4 "Chia dữ liệu theo buổi"     "python scripts/make_splits.py"
Step 5 "Tiền xử lý (Lab + mask)"    "python scripts/preprocess.py"
Step 6 "Train + đánh giá (chạy thử)" "python scripts/train.py --epochs $Epochs --image-size $ImageSize --name smoke_test"

Write-Host ""
Write-Host "XONG! Toàn bộ quy trình chạy thông." -ForegroundColor Green
Write-Host "Kết quả: outputs\runs\smoke_test\  và  outputs\figures\mask_check.jpg"