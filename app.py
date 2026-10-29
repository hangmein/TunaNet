"""TunaNet web server. Run: python app.py --model outputs/runs/<name>/best.pt"""
import argparse
import base64
import io
import os
from pathlib import Path
import threading
import time
import warnings

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from werkzeug.exceptions import RequestEntityTooLarge

from scripts.predict import load_model, predict_one

ROOT = Path(__file__).resolve().parent
MAX_FILE = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


class Predictor:
    def __init__(self, model_path):
        self.path = Path(model_path)
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.classes = []
        self.error = None
        self.lock = threading.Lock()
        if self.path.is_file():
            try:
                self.model, self.classes, self.size = load_model(self.path, self.device)
            except Exception:
                self.error = 'Không thể nạp checkpoint. Kiểm tra file trọng số và khởi động lại server.'
        else:
            self.error = 'Chưa có trọng số mô hình. Thêm best.pt rồi khởi động server với --model đường_dẫn/best.pt.'

    def status(self):
        return {'ready': self.model is not None, 'message': self.error,
                'device': self.device.upper(), 'classes': self.classes,
                'model': self.path.name}

    def predict(self, bgr):
        with self.lock:
            prob, box, crop, mask, ratio = predict_one(self.model, bgr, self.size, self.device)
        if not np.isfinite(prob).all():
            raise ValueError('Model returned non-finite probabilities')
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = crop.copy()
        cv2.drawContours(overlay, contours, -1, (220, 220, 40), 2)
        ok, encoded = cv2.imencode('.jpg', overlay)
        if not ok:
            raise ValueError('Cannot encode preview')
        k = int(prob.argmax())
        return {'prediction': self.classes[k], 'confidence': float(prob[k]),
                'probabilities': [{'label': c, 'probability': float(p)} for c, p in zip(self.classes, prob)],
                'meat_ratio': ratio, 'box': box,
                'processed_image': 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii'),
                'warning': ('Vùng thịt nhận diện bất thường. Hãy kiểm tra ảnh và vùng viền được đánh dấu.'
                            if ratio < 0.15 or ratio > 0.97 else None)}


def decode_image(raw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in {'JPEG', 'PNG', 'WEBP', 'BMP'}:
                    raise ValueError('Chỉ hỗ trợ ảnh JPG, PNG, WEBP hoặc BMP.')
                if im.width * im.height > MAX_PIXELS or min(im.size) < 32 or max(im.size) / min(im.size) > 20:
                    raise ValueError('Ảnh phải có mỗi cạnh ít nhất 32 px, tối đa 20 megapixel và tỉ lệ cạnh không quá 20:1.')
                im.load()
                rgb = np.array(ImageOps.exif_transpose(im).convert('RGB'))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('Không đọc được ảnh. Hãy chọn một file ảnh hợp lệ, không bị hỏng.') from exc


def create_app(model_path=None, predictor=None):
    app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
    # Waitress does not enable Flask's debug template reloader by default.
    # Recheck HTML on each request so a running server cannot mix old HTML
    # with freshly edited CSS/JS.
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE + 1024 * 1024
    service = predictor if predictor is not None else Predictor(
        model_path or os.environ.get('TUNANET_MODEL', 'outputs/runs/wb_on_rev/best.pt'))
    app.extensions['predictor'] = service

    @app.get('/')
    def index():
        versions = {name: (Path(app.static_folder) / name).stat().st_mtime_ns
                    for name in ('style.css', 'app.js')}
        return render_template('index.html', asset_versions=versions)

    @app.get('/api/health')
    def health():
        return jsonify(service.status())

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return jsonify(error='Ảnh quá lớn. Dung lượng tối đa là 10 MB.'), 413

    @app.post('/api/predict')
    def predict():
        upload = request.files.get('image')
        if upload is None or not upload.filename:
            return jsonify(error='Vui lòng chọn một ảnh.'), 400
        raw = upload.read(MAX_FILE + 1)
        if len(raw) > MAX_FILE:
            return too_large(None)
        try:
            bgr = decode_image(raw)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if not service.status()['ready']:
            return jsonify(error=service.status()['message']), 503
        start = time.perf_counter()
        try:
            result = service.predict(bgr)
        except Exception:
            app.logger.exception('TunaNet inference failed')
            return jsonify(error='Không thể xử lý ảnh. Hãy thử ảnh khác hoặc kiểm tra log server.'), 500
        result['elapsed_ms'] = round((time.perf_counter() - start) * 1000)
        return jsonify(result)

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TunaNet web application')
    parser.add_argument('--model', default=os.environ.get('TUNANET_MODEL', 'outputs/runs/wb_on_rev/best.pt'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    torch.set_num_threads(8)
    from waitress import serve
    application = create_app(args.model)
    print(f'TunaNet: http://{args.host}:{args.port}', flush=True)
    state = application.extensions['predictor'].status()
    print(f"Model ready: {state['ready']} | Device: {state['device']}", flush=True)
    serve(application, host=args.host, port=args.port, threads=4,
          max_request_body_size=MAX_FILE + 2 * 1024 * 1024)
