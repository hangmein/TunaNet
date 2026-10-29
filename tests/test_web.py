"""HTTP contract and real inference plumbing; synthetic weights are never shipped."""
import io
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch
from jinja2 import FileSystemLoader

from app import create_app, decode_image
from scripts.predict import predict_one
from src.models.tunanet import TunaNet


def sample():
    img = np.full((100, 140, 3), 230, dtype=np.uint8)
    cv2.ellipse(img, (70, 50), (40, 30), 0, 0, 360, (70, 65, 170), -1)
    return cv2.imencode('.png', img)[1].tobytes()


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / 'synthetic-test-only.pt'
        torch.manual_seed(42)
        torch.save({'model': TunaNet(3).state_dict(), 'classes': ['Grade-A', 'Grade-B', 'Grade-C'],
                    'args': {'image_size': 64}}, cls.path)
        cls.app = create_app(cls.path)
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def post(self, data, filename='ảnh cá.png', client=None):
        return (client or self.client).post('/api/predict', data={'image': (io.BytesIO(data), filename)})

    def test_page_and_assets(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        for path in ['/static/style.css', '/static/app.js']:
            self.assertEqual(self.client.get(path).status_code, 200)
        self.assertTrue(self.client.get('/api/health').json['ready'])

    def test_actual_pipeline_matches_cli(self):
        result = self.post(sample())
        self.assertEqual(result.status_code, 200)
        payload = result.json
        service = self.app.extensions['predictor']
        expected = predict_one(service.model, decode_image(sample()), service.size, service.device)[0]
        np.testing.assert_allclose([r['probability'] for r in payload['probabilities']], expected)
        self.assertAlmostEqual(sum(r['probability'] for r in payload['probabilities']), 1, places=5)
        self.assertEqual(payload['prediction'], service.classes[int(expected.argmax())])
        self.assertTrue(payload['processed_image'].startswith('data:image/jpeg;base64,'))

    def test_template_updates_without_restart(self):
        app = create_app(self.path)
        template_dir = Path(self.tmp.name) / 'templates'
        template_dir.mkdir(exist_ok=True)
        template = template_dir / 'index.html'
        template.write_text('old layout', encoding='utf-8')
        app.jinja_loader = FileSystemLoader(template_dir)
        client = app.test_client()
        self.assertEqual(client.get('/').text, 'old layout')
        previous = template.stat().st_mtime
        template.write_text('new layout', encoding='utf-8')
        os.utime(template, (previous + 2, previous + 2))
        self.assertEqual(client.get('/').text, 'new layout')

    def test_page_uses_current_layout_and_versioned_assets(self):
        page = self.client.get('/').text
        self.assertIn('Không gian làm việc', page)
        self.assertIn('aria-label="Thanh điều hướng"', page)
        self.assertNotIn('VISION LAB', page)
        self.assertIn('/static/style.css?v=', page)
        self.assertIn('/static/app.js?v=', page)

    def test_missing_model(self):
        client = create_app(Path(self.tmp.name) / 'missing.pt').test_client()
        self.assertFalse(client.get('/api/health').json['ready'])
        self.assertEqual(self.post(sample(), client=client).status_code, 503)

    def test_corrupt_checkpoint(self):
        path = Path(self.tmp.name) / 'broken.pt'
        path.write_bytes(b'broken')
        client = create_app(path).test_client()
        self.assertFalse(client.get('/api/health').json['ready'])
        self.assertEqual(self.post(sample(), client=client).status_code, 503)

    def test_invalid_uploads(self):
        self.assertEqual(self.client.post('/api/predict').status_code, 400)
        self.assertEqual(self.post(b'not an image').status_code, 400)
        self.assertEqual(self.post(b'').status_code, 400)
        self.assertEqual(self.post(b'x' * (10 * 1024 * 1024 + 1)).status_code, 413)
        tiny = cv2.imencode('.png', np.zeros((1, 1, 3), dtype=np.uint8))[1].tobytes()
        self.assertEqual(self.post(tiny).status_code, 400)

    def test_blank_image_warning(self):
        raw = cv2.imencode('.png', np.full((80, 80, 3), 255, dtype=np.uint8))[1].tobytes()
        response = self.post(raw)
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.json['warning'])


if __name__ == '__main__':
    unittest.main()
