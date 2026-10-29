'use strict';
const $ = id => document.getElementById(id);
let selected = null;
let previewURL = null;
let busy = false;
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
function resetResult() {
  $('result-state').classList.remove('failed');
  $('result-content').hidden = true;
  $('result-empty').hidden = false;
  $('result-empty').querySelector('h3').textContent = 'Chưa có kết quả';
  $('result-empty').querySelector('p').textContent = 'Chọn ảnh và nhấn “Phân tích ảnh”. Kết quả dự đoán sẽ xuất hiện tại đây.';
  $('result-state').textContent = selected ? 'Chờ phân tích' : 'Chờ ảnh';
}
function chooseFile(file) {
  if (busy || !file) return;
  showError('');
  selected = null;
  $('submit').disabled = true;
  $('file-details').hidden = true;
  $('file-hint').hidden = false;
  $('preview').hidden = true;
  $('empty-upload').hidden = false;
  if (previewURL) { URL.revokeObjectURL(previewURL); previewURL = null; }
  resetResult();
  if (!/\.(jpe?g|png|webp|bmp)$/i.test(file.name)) return showError('Vui lòng chọn ảnh JPG, PNG, WEBP hoặc BMP.');
  if (!file.size || file.size > 10 * 1024 * 1024) return showError('Ảnh phải có dung lượng từ 1 byte đến 10 MB.');
  selected = file;
  previewURL = URL.createObjectURL(file);
  $('preview').src = previewURL;
  $('preview').hidden = false;
  $('empty-upload').hidden = true;
  $('file-details').hidden = false;
  $('file-hint').hidden = true;
  $('file-name').textContent = file.name;
  $('file-size').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · Sẵn sàng gửi lên server`;
  $('submit').disabled = false;
  resetResult();
}
$('preview').addEventListener('error', () => {
  selected = null; $('submit').disabled = true; $('preview').hidden = true;
  $('empty-upload').hidden = false; $('file-details').hidden = true;
  $('file-hint').hidden = false;
  resetResult(); showError('Trình duyệt không đọc được ảnh này. Hãy chọn ảnh khác.');
});
['choose', 'replace'].forEach(id => $(id).addEventListener('click', () => { $('file-input').value = ''; $('file-input').click(); }));
$('file-input').addEventListener('change', event => chooseFile(event.target.files[0]));
$('reset').addEventListener('click', () => {
  if (busy) return;
  selected = null;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  $('preview').hidden = true;
  $('preview').removeAttribute('src');
  $('empty-upload').hidden = false;
  $('file-details').hidden = true;
  $('file-hint').hidden = false;
  $('file-input').value = '';
  $('submit').disabled = true;
  $('processed-image').removeAttribute('src');
  $('result-content').querySelector('details').open = false;
  resetResult(); showError(''); health();
});
const zone = $('drop-zone');
['dragenter', 'dragover'].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); if (!busy) zone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove('dragging'); }));
zone.addEventListener('drop', event => { if (event.dataTransfer.files.length !== 1) showError('Vui lòng chọn mỗi lần một ảnh.'); else chooseFile(event.dataTransfer.files[0]); });
async function health() {
  try {
    const response = await fetch('/api/health');
    if (!response.ok) throw new Error();
    const data = await response.json();
    $('connection').replaceChildren(document.createElement('i'), document.createTextNode(data.ready ? `Mô hình sẵn sàng · ${data.device}` : 'Chưa có mô hình'));
    $('connection').classList.toggle('ready', data.ready);
    $('model-notice').textContent = data.message || '';
    $('model-notice').hidden = data.ready;
  } catch {
    $('connection').textContent = 'Mất kết nối server';
    $('connection').classList.remove('ready');
    $('model-notice').textContent = 'Không kết nối được server. Kiểm tra server đang chạy rồi tải lại trang.';
    $('model-notice').hidden = false;
  }
}
$('upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (!selected || busy) return;
  busy = true; showError(''); resetResult();
  $('results-panel').hidden = false;
  document.body.classList.add('busy');
  ['submit', 'choose', 'replace', 'file-input', 'reset'].forEach(id => $(id).disabled = true);
  $('results-panel').setAttribute('aria-busy', 'true');
  $('submit-label').textContent = 'Đang phân tích…';
  $('result-state').textContent = 'Đang xử lý';
  $('result-empty').querySelector('h3').textContent = 'TunaNet đang phân tích';
  $('result-empty').querySelector('p').textContent = 'Ảnh đang được gửi và xử lý trên server. Vui lòng chờ.';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const body = new FormData(); body.append('image', selected);
    const response = await fetch('/api/predict', { method: 'POST', body, signal: controller.signal });
    let data;
    try { data = await response.json(); } catch { throw new Error('Server trả về phản hồi không hợp lệ. Hãy thử lại.'); }
    if (!response.ok) throw new Error(data.error || 'Không thể phân tích ảnh.');
    $('prediction').textContent = data.prediction;
    $('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    $('elapsed').textContent = `${(data.elapsed_ms / 1000).toFixed(2)} giây`;
    $('probabilities').replaceChildren();
    for (const item of data.probabilities) {
      const row = document.createElement('tr');
      const label = document.createElement('td'); label.textContent = item.label;
      const value = document.createElement('td'); value.textContent = `${(item.probability * 100).toFixed(1)}%`;
      if (item.label === data.prediction) {
        row.className = 'predicted';
        const badge = document.createElement('small'); badge.textContent = 'Dự đoán'; label.append(badge);
      }
      row.append(label, value); $('probabilities').append(row);
    }
    $('processed-image').src = data.processed_image;
    $('meat-ratio').textContent = `${(data.meat_ratio * 100).toFixed(1)}% diện tích`;
    $('warning').textContent = data.warning || ''; $('warning').hidden = !data.warning;
    $('result-content').hidden = false; $('result-empty').hidden = true;
    $('result-state').textContent = 'Hoàn tất';
  } catch (error) {
    showError(error.name === 'AbortError' ? 'Xử lý quá thời gian chờ. Vui lòng kiểm tra server và thử lại.' : (error instanceof TypeError ? 'Mất kết nối server. Vui lòng kiểm tra và thử lại.' : error.message));
    resetResult(); $('result-state').textContent = 'Không thành công';
    $('result-state').classList.add('failed');
  } finally {
    clearTimeout(timeout); busy = false; document.body.classList.remove('busy');
    ['choose', 'replace', 'file-input', 'reset'].forEach(id => $(id).disabled = false);
    $('submit').disabled = !selected;
    $('submit-label').textContent = 'Phân tích ảnh';
    $('results-panel').setAttribute('aria-busy', 'false');
    health();
  }
});
health();
