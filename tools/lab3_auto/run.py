"""Resumable, CPU-only auto-annotation and genuine CVAT dataset export."""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from urllib.parse import urlencode, quote
import zipfile

import docker
import numpy as np
from PIL import Image
import requests

ROOT = Path('/workspace')
WORK = ROOT / '.lab3-auto'
VERSION = 'lab3-auto-v1'
SAM = 'http://nuclio-nuclio-pth-facebookresearch-sam-vit-b-cpu:8080'
CONFIG = {'version': VERSION, 'yolo': 'yolo11n.pt', 'imgsz': 1280, 'conf': .25, 'iou': .5,
          'semantic': 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024', 'sam': 'SAM ViT-B CPU', 'threads': 4}


def log(message):
    print(datetime.now().strftime('%H:%M:%S'), message, flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def encode(mask):
    h, w = mask.shape
    pixels = mask.reshape(-1).astype(np.uint8)
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    runs = np.diff(np.r_[0, changes, pixels.size]).tolist()
    if pixels[0]:
        runs.insert(0, 0)
    return runs + [0, 0, w-1, h-1]


def decode(points):
    x0, y0, x1, y1 = map(int, points[-4:])
    runs = np.asarray(points[:-4], dtype=np.int64)
    assert np.all(runs >= 0) and runs.sum() == (x1-x0+1)*(y1-y0+1)
    return np.repeat(np.arange(len(runs)) % 2, runs).reshape(y1-y0+1, x1-x0+1).astype(bool)


def mask_signature(s):
    return fingerprint({'frame': s['frame'], 'label_id': s['label_id'],
                        'points': [int(v) for v in s['points']], 'type': s['type'], 'source': s['source']})


class CVAT:
    def __init__(self):
        self.docker = docker.from_env()
        self.server = self.docker.containers.get('cvat_server')
        self.put({'bridge.py': (ROOT / 'tools/lab3_auto/bridge.py').read_bytes()})

    def put(self, files):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w') as tar:
            for name, data in files.items():
                info = tarfile.TarInfo('lab3-auto/' + name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        self.server.put_archive('/tmp', buf.getvalue())

    def call(self, path, method='GET', data=None, files=None, download=False):
        arg = {'path': path, 'method': method, 'data': data or {}, 'download': download}
        if files:
            self.put({'images/' + p.name: p.read_bytes() for p in files})
            arg['files'] = ['/tmp/lab3-auto/images/' + p.name for p in files]
        self.put({'request.json': json.dumps(arg).encode()})
        # Mutating calls are never retried blindly: the checkpoint/reconciliation handles them.
        attempts = 3 if method == 'GET' else 1
        for attempt in range(attempts):
            result = self.server.exec_run(['python', '/tmp/lab3-auto/bridge.py', '/tmp/lab3-auto/request.json'], workdir='/home/django')
            output = result.output.decode(errors='replace')
            if result.exit_code == 0 and 'LAB3_RESULT=' in output:
                body = json.loads(output.rsplit('LAB3_RESULT=', 1)[1])
                if body['status'] < 400:
                    return body['data']
                error = f"CVAT {method} {path}: {body}"
            else:
                error = output[-5000:]
            if attempt + 1 < attempts:
                time.sleep(2*(attempt+1))
        raise RuntimeError(error)

    def wait(self, rq_id):
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            result = self.call('/api/requests/' + quote(rq_id, safe=''))
            if result['status'] == 'finished':
                return result
            if result['status'] in ('failed', 'canceled'):
                raise RuntimeError(str(result))
            time.sleep(3)
        raise TimeoutError(f'CVAT request timed out: {rq_id}')

    def get_file(self, path):
        stream, _ = self.server.get_archive(path)
        with tarfile.open(fileobj=io.BytesIO(b''.join(stream))) as tar:
            return tar.extractfile(tar.getmembers()[0]).read()


class Models:
    def __init__(self):
        import torch
        torch.set_num_threads(4)
        self.torch = torch
        self.yolo = None
        self.semantic = None
        self.processor = None

    def detect(self, image, classes):
        if self.yolo is None:
            from ultralytics import YOLO
            p = Path('/cache/models/yolo11n.pt')
            p.parent.mkdir(parents=True, exist_ok=True)
            self.yolo = YOLO(str(p))
        result = self.yolo.predict(image, imgsz=1280, conf=.25, iou=.5, device='cpu', verbose=False)[0]
        found = []
        for box in result.boxes:
            label = result.names[int(box.cls.item())]
            if label in classes:
                found.append({'label': label, 'confidence': float(box.conf.item()), 'bbox': box.xyxy[0].tolist()})
        return sorted(found, key=lambda item: -item['confidence'])

    def segment(self, image):
        if self.semantic is None:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            self.processor = AutoImageProcessor.from_pretrained(CONFIG['semantic'], use_fast=False)
            self.semantic = AutoModelForSemanticSegmentation.from_pretrained(CONFIG['semantic']).eval()
        with self.torch.inference_mode():
            inputs = self.processor(images=image, return_tensors='pt')
            logits = self.semantic(**inputs).logits
            logits = self.torch.nn.functional.interpolate(logits, size=(image.height, image.width), mode='bilinear', align_corners=False)
            confidence, labels = logits.softmax(dim=1).max(dim=1)
        names = {int(k): v for k, v in self.semantic.config.id2label.items()}
        return labels[0].numpy(), confidence[0].numpy(), names

    def sam(self, raw, box):
        error = None
        for attempt in range(3):
            try:
                response = requests.post(SAM, json={'image': raw, 'obj_bbox': box, 'pos_points': [], 'neg_points': []}, timeout=180)
                response.raise_for_status()
                return decode(response.json()['shapes'][0]['points'])
            except (requests.RequestException, ValueError) as exc:
                error = exc
                time.sleep(2*(attempt+1))
        raise RuntimeError(f'SAM failed: {error}')


def predict(models, path, info, taxonomy, cache):
    key = fingerprint({'image': sha(path), 'type': info['type'], 'taxonomy': taxonomy, 'config': CONFIG})
    dest = cache / (path.stem + '.json')
    if dest.exists():
        previous = json.loads(dest.read_text())
        if previous.get('key') == key:
            log('Cache: ' + path.name)
            return previous
    started = time.monotonic()
    image = Image.open(path).convert('RGB')
    shape = (image.height, image.width)
    classes = taxonomy['classes']
    stuff = set(taxonomy.get('stuff', []))
    masks = []
    overlaps = np.zeros(shape, dtype=bool)
    occupied = np.zeros(shape, dtype=bool)
    if info['type'] in ('instance', 'panoptic'):
        targets = set(classes) - stuff
        detections = models.detect(image, targets)
        log(f'{path.name}: {len(detections)} detections → SAM')
        raw = base64.b64encode(path.read_bytes()).decode()
        for detection in detections:
            mask = models.sam(raw, detection['bbox'])
            assert mask.shape == shape
            overlaps |= occupied & mask
            if info['type'] == 'panoptic':
                mask &= ~occupied
            if not mask.any():
                continue
            occupied |= mask
            masks.append({**detection, 'points': encode(mask), 'pixels': int(mask.sum())})
    if info['type'] in ('semantic', 'panoptic'):
        labels, confidence, mapping = models.segment(image)
        targets = set(classes) if info['type'] == 'semantic' else stuff
        for label_id, label in mapping.items():
            if label not in targets:
                continue
            mask = labels == label_id
            if info['type'] == 'panoptic':
                overlaps |= mask & occupied
                mask &= ~occupied
            if mask.any():
                occupied |= mask
                masks.append({'label': label, 'confidence': float(confidence[mask].mean()), 'points': encode(mask), 'pixels': int(mask.sum())})
    warnings = []
    low = sum(m['confidence'] < .5 for m in masks)
    if low:
        warnings.append(f'{low} masks have confidence below 0.5 (model-specific, not calibrated).')
    if not masks:
        warnings.append('No masks detected; do not interpret this as no objects present.')
    result = {'key': key, 'image': path.name, 'width': image.width, 'height': image.height, 'masks': masks,
              'seconds': round(time.monotonic()-started, 2), 'unlabeled_pixels': int((~occupied).sum()),
              'overlap_pixels_before_resolution': int(overlaps.sum()), 'low_confidence_masks': low, 'warnings': warnings}
    write_json(dest, result)
    colors = taxonomy.get('colors', {})
    overlay = np.asarray(image).copy()
    for index, m in enumerate(masks):
        mask = decode(m['points'])
        color = np.asarray(colors.get(m['label'], [255, 0, 255]))
        overlay[mask] = (overlay[mask]*.55 + color*.45).astype(np.uint8)
    preview = ROOT / 'automation_previews' / cache.name
    preview.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(preview / (path.stem + '.png'))
    return result


def inspect(task, path):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location('lab_inspect', ROOT / 'scripts/inspect_submissions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.inspect_task(task, path, ROOT)
    if result['errors']:
        raise RuntimeError('ZIP validation: ' + '; '.join(result['errors']))
    return result


def process(name, info, cvat, models, state, save):
    folder = ROOT / 'data' / info['path']
    images = sorted(p for p in (folder / 'images').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    taxonomy = json.loads((folder / 'classes.json').read_text())
    labels = json.loads((folder / 'cvat-labels.json').read_text())
    row = state['tasks'].setdefault(name, {})
    input_hash = fingerprint({p.name: sha(p) for p in images})
    if row.get('input_hash') not in (None, input_hash):
        raise RuntimeError('Input images changed; refusing to reuse an existing task')
    row['input_hash'] = input_hash
    output = ROOT / 'submissions' / (name + '.zip')
    if row.get('status') == 'complete' and output.exists() and sha(output) == row.get('zip_sha256'):
        current = cvat.call(f"/api/tasks/{row['task_id']}/annotations")
        if fingerprint(current) != row.get('annotations_hash'):
            raise RuntimeError('Annotations changed after export; preserving manual edits and old ZIP')
        inspect(name, output)
        log(name + ': already complete, verified; no changes')
        return
    started = time.monotonic()
    if not row.get('task_id'):
        task_name = row.get('task_prefix', 'auto_') + name
        candidates = cvat.call('/api/tasks?' + urlencode({'search': task_name, 'page_size': 100}))['results']
        matching = [x for x in candidates if x['name'] == task_name and x.get('bug_tracker') == 'https://localhost/lab3-auto/' + name]
        if len(matching) > 1:
            raise RuntimeError('Multiple matching managed tasks')
        task = matching[0] if matching else cvat.call('/api/tasks', 'POST', {'name': task_name, 'labels': labels, 'segment_size': 1000, 'overlap': 0, 'bug_tracker': 'https://localhost/lab3-auto/' + name})
        row['task_id'] = task['id']
        row['status'] = 'created'
        save()
    task_id = row['task_id']
    task = cvat.call(f'/api/tasks/{task_id}')
    if not task.get('size'):
        if row.get('upload_rq_id'):
            cvat.wait(row['upload_rq_id'])
        else:
            result = cvat.call(f'/api/tasks/{task_id}/data', 'POST', {'image_quality': 100, 'sorting_method': 'lexicographical', 'use_cache': True}, files=images)
            row['upload_rq_id'] = result.get('rq_id')
            save()
            if row['upload_rq_id']:
                cvat.wait(row['upload_rq_id'])
    meta = cvat.call(f'/api/tasks/{task_id}/data/meta')
    frame_names = [Path(f['name']).name for f in meta['frames']]
    if sorted(frame_names) != sorted(p.name for p in images):
        raise RuntimeError('CVAT frame names do not match manifest inputs')
    jobs = cvat.call('/api/jobs?' + urlencode({'task_id': task_id, 'page_size': 100}))['results']
    if len(jobs) != 1:
        raise RuntimeError('Expected exactly one job')
    row['job_id'] = jobs[0]['id']
    labellist = cvat.call('/api/labels?' + urlencode({'task_id': task_id, 'page_size': 100}))['results']
    ids = {x['name']: x['id'] for x in labellist}
    if set(ids) != set(taxonomy['classes']):
        raise RuntimeError('CVAT taxonomy mismatch')
    existing = cvat.call(f'/api/tasks/{task_id}/annotations')
    if any(s.get('source') != 'auto' for s in existing.get('shapes', [])) or existing.get('tracks') or existing.get('tags'):
        raise RuntimeError('Task contains non-tool annotations; refusing replacement')
    if row.get('annotations_hash') and fingerprint(existing) != row['annotations_hash']:
        raise RuntimeError('Task was changed externally; refusing replacement')
    predictions = []
    shapes = []
    for frame, filename in enumerate(frame_names):
        if row.get('prediction_override_dir'):
            result = json.loads((ROOT / row['prediction_override_dir'] / (Path(filename).stem + '.json')).read_text())
            if result.get('image_sha256') != sha(folder / 'images' / filename):
                raise RuntimeError('Selected improved prediction does not match the input image')
        else:
            result = predict(models, folder / 'images' / filename, info, taxonomy, WORK / 'predictions' / name)
        predictions.append({k: v for k, v in result.items() if k != 'masks'})
        predictions[-1]['mask_count'] = len(result['masks'])
        predictions[-1]['classes'] = dict(Counter(m['label'] for m in result['masks']))
        for m in result['masks']:
            decoded = decode(m['points'])
            assert decoded.shape == (result['height'], result['width'])
            shapes.append({'type': 'mask', 'frame': frame, 'label_id': ids[m['label']], 'points': m['points'],
                           'occluded': False, 'outside': False, 'z_order': 0, 'rotation': 0, 'attributes': [], 'source': 'auto'})
    if not shapes:
        raise RuntimeError('No annotations produced; refusing empty submission')
    payload = {'version': existing.get('version', 0), 'tags': [], 'tracks': [], 'shapes': shapes}
    cvat.call(f'/api/tasks/{task_id}/annotations', 'PUT', payload)
    stored = cvat.call(f'/api/tasks/{task_id}/annotations')
    if len(stored['shapes']) != len(shapes):
        raise RuntimeError('Annotation count differs after save')
    # CVAT may return shapes in a different order and serializes RLE integers as floats.
    if Counter(map(mask_signature, stored['shapes'])) != Counter(map(mask_signature, shapes)):
        raise RuntimeError('Stored annotation differs from generated mask')
    row.update(status='annotated', annotations_hash=fingerprint(stored), annotation_count=len(shapes), images=predictions)
    save()
    fmt = 'Segmentation mask 1.1' if info['type'] == 'semantic' else 'COCO 1.0'
    export = cvat.call(f'/api/tasks/{task_id}/dataset/export?' + urlencode({'format': fmt, 'save_images': 'false', 'location': 'local'}), 'POST')
    row['export_rq_id'] = export['rq_id']
    save()
    result = cvat.wait(export['rq_id'])
    exported = cvat.call(result['result_url'], download=True)
    data = cvat.get_file(exported['file'])
    if not zipfile.is_zipfile(io.BytesIO(data)):
        raise RuntimeError('CVAT did not return a ZIP')
    candidate = WORK / 'exports' / (name + '.zip')
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(data)
    qc = inspect(name, candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        backup = WORK / 'backups' / (name + '-' + str(time.time_ns()) + '.zip')
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(output.read_bytes())
    output.write_bytes(data)
    row.update(status='complete', format=fmt, zip_sha256=sha(output), zip_bytes=len(data), seconds=round(time.monotonic()-started, 2), qc=qc, error=None)
    save()
    log(f'{name}: saved {len(shapes)} masks, ZIP {len(data):,} bytes')


def report(state):
    completed = [r for r in state['tasks'].values() if r.get('status') == 'complete']
    lines = ['# Lab 3 — Nhật ký chạy tự động', '', f"Cập nhật UTC+7: {datetime.now(timezone(timedelta(hours=7))).isoformat()}", '',
             f"Đã hoàn tất {len(completed)} task, {sum(len(r.get('images', [])) for r in completed)} ảnh, {sum(r.get('annotation_count', 0) for r in completed)} mask.", '',
             'Dữ liệu được tạo tự động bằng YOLO + SAM + SegFormer. Kiểm tra cấu trúc không chứng minh mask đúng về nội dung.',
             'Không thực hiện hoặc xác nhận bước tự vẽ/kiểm tra thủ công thay cho học viên. Không push GitHub hoặc nộp LMS.', '',
             '## Cấu hình', '', '```json', json.dumps(CONFIG, indent=2), '```', '',
             '## Bảo toàn Easy', '', f"Easy không thay đổi: {state.get('easy_unchanged', 'chưa kiểm tra cuối lượt')}", '',
             '## Task và ZIP', '', '| Task | Trạng thái | Task / Job | Masks | Giây | ZIP bytes |', '|---|---|---|---:|---:|---:|']
    for name, row in state['tasks'].items():
        lines.append(f"| {name} | {row.get('status')} | {row.get('task_id','')} / {row.get('job_id','')} | {row.get('annotation_count','')} | {row.get('seconds','')} | {row.get('zip_bytes','')} |")
    for name, row in state['tasks'].items():
        lines += ['', '## ' + name, '', f"Task: http://localhost:8080/tasks/{row.get('task_id','')}", f"ZIP: `submissions/{name}.zip`", f"SHA-256: `{row.get('zip_sha256','chưa có')}`"]
        if row.get('prediction_model'):
            lines += [f"Mô hình của phiên bản hiện tại: {row['prediction_model']}"]
        if row.get('error'):
            lines += ['', 'Lỗi thực tế:', '```text', row['error'], '```']
        for item in row.get('images', []):
            lines += ['', f"- `{item['image']}`: {item['mask_count']} masks; {item['seconds']} giây; chưa gán nhãn {item['unlabeled_pixels']} pixel; chồng lấn trước xử lý {item['overlap_pixels_before_resolution']} pixel; confidence thấp {item['low_confidence_masks']} masks.", f"  Lớp: {json.dumps(item['classes'], ensure_ascii=False)}"]
            lines.extend('  - ' + warning for warning in item['warnings'])
        lines.extend('- Cảnh báo kiểm ZIP: ' + w for w in row.get('qc', {}).get('warnings', []))
    visual = WORK / 'visual_review.json'
    if visual.exists():
        lines += ['', '## Quan sát overlay bởi trợ lý (không phải kiểm thủ công của học viên)', '']
        lines.extend('- ' + item for item in json.loads(visual.read_text()))
    verification = WORK / 'verification.json'
    if verification.exists():
        lines += ['', '## Kiểm thử tool', '', '```json', verification.read_text(), '```']
    runtime = WORK / 'runtime.json'
    if runtime.exists():
        lines += ['', '## Phiên bản thư viện đã chạy', '', '```json', runtime.read_text(), '```']
    lines += ['', '## Giới hạn', '', '- Vùng không thuộc taxonomy được để trống, không bịa nhãn để đủ coverage.',
              '- Vật nhỏ, biên mảnh, vật bị che và kính/khe cần kiểm trực quan; không tự suy ra phần khuất.',
              '- Overlay nằm trong `automation_previews/`. Báo cáo này không thay báo cáo cá nhân theo mẫu lớp.']
    (ROOT / 'AUTOMATION_RUN_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='*')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    import importlib.metadata
    write_json(WORK / 'runtime.json', {p: importlib.metadata.version(p) for p in ('ultralytics', 'transformers', 'torch', 'torchvision', 'docker', 'requests')})
    import fcntl
    lock = (WORK / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = WORK / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'tasks': {}, 'config': CONFIG}
    if state.get('config') != CONFIG:
        raise RuntimeError('Configuration changed; do not silently reuse task state')
    cvat = CVAT()
    easy_path = ROOT / 'submissions/easy_semantic.zip'
    baseline = {'zip': sha(easy_path), 'annotations': fingerprint(cvat.call('/api/tasks/3/annotations'))}
    if state.get('easy_baseline') and state['easy_baseline'] != baseline:
        raise RuntimeError('Easy changed since initial run; inspect before continuing')
    state['easy_baseline'] = baseline
    def save():
        write_json(state_path, state)
        report(state)
    save()
    if args.check_only:
        log('Preflight OK: bridge authenticated locally; Easy baseline captured')
        return
    registry = json.loads((ROOT / 'data/manifest.json').read_text())['tasks']
    targets = args.tasks or [n for n in registry if n != 'easy_semantic']
    if any(n == 'easy_semantic' or n not in registry for n in targets):
        raise ValueError('Only the eight remaining manifest tasks are supported')
    models = Models()
    errors = []
    for name in targets:
        try:
            log('START ' + name)
            process(name, registry[name], cvat, models, state, save)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row = state['tasks'].setdefault(name, {})
            row.update(status='failed', error=str(exc))
            errors.append(name)
            save()
    after = {'zip': sha(easy_path), 'annotations': fingerprint(cvat.call('/api/tasks/3/annotations'))}
    state['easy_unchanged'] = after == baseline
    save()
    if not state['easy_unchanged']:
        raise RuntimeError('Easy preservation check FAILED')
    log('FINISHED; Easy unchanged; failed tasks: ' + str(errors))
    sys.exit(bool(errors))


if __name__ == '__main__':
    main()
