"""Evaluate stronger, image-only predictors; promote only improved CVAT exports.

Released references are read only by evaluate(), never by either predictor.
"""
import argparse
import base64
from collections import Counter
import gc
import io
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
import torch

import run as runner

ROOT = runner.ROOT
WORK = ROOT / '.lab3-improve'
REGISTRY = json.loads((ROOT/'data/manifest.json').read_text())['tasks']
MODEL = 'facebook/mask2former-swin-small-coco-panoptic'
M2F_VARIANT = 'mask2former-small-safe'
ALIASES = {'building-other-merged': 'building', 'tree-merged': 'vegetation',
           'sky-other-merged': 'sky', 'pavement-merged': 'sidewalk'}
TASKS = ('medium_instance', 'hard_panoptic')
VARIANTS = {'medium_instance': ['yolo11l-sam', M2F_VARIANT, 'yolo11l-tiled-sam'],
            'hard_panoptic': [M2F_VARIANT, 'tiled-segformer', 'tiled-mask2former']}
for _task in TASKS:
    VARIANTS[_task] += ['ensemble-050', 'ensemble-065']
VARIANTS['hard_panoptic'] += ['ensemble-stufffill-050']
sys.path.insert(0, str(ROOT))
import lab_utils as metrics


def taxonomy(task):
    return json.loads((ROOT/'data'/REGISTRY[task]['path']/'classes.json').read_text())


def files(task):
    return sorted((ROOT/'data'/REGISTRY[task]['path']/'images').glob('*.jpg'))


def result_path(variant, task, image):
    return WORK/'variants'/variant/task/(image.stem+'.json')


def save_prediction(variant, task, image, masks, elapsed):
    rgb = np.asarray(Image.open(image).convert('RGB'))
    h, w = rgb.shape[:2]
    occupied = np.zeros((h, w), bool)
    overlap = np.zeros((h, w), bool)
    for m in masks:
        mask = runner.decode(m['points'])
        assert mask.shape == (h, w)
        overlap |= occupied & mask
        occupied |= mask
    result = {'image': image.name, 'image_sha256': runner.sha(image), 'width': w, 'height': h,
              'masks': masks, 'seconds': round(elapsed, 2), 'unlabeled_pixels': int((~occupied).sum()),
              'overlap_pixels_before_resolution': int(overlap.sum()),
              'low_confidence_masks': sum(m['confidence'] < .5 for m in masks), 'warnings': []}
    runner.write_json(result_path(variant, task, image), result)
    colors = taxonomy(task)['colors']
    overlay = rgb.copy()
    for m in masks:
        mask = runner.decode(m['points'])
        overlay[mask] = (overlay[mask]*.55+np.asarray(colors[m['label']])*.45).astype(np.uint8)
    out = WORK/'variants'/variant/'overlays'/task
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out/(image.stem+'.png'))


def cached(variant, task, image):
    p = result_path(variant, task, image)
    return p.exists() and json.loads(p.read_text()).get('image_sha256') == runner.sha(image)


def predict_yolo():
    variant = 'yolo11l-sam'
    from ultralytics import YOLO
    models = runner.Models()
    models.yolo = YOLO('/cache/models/yolo11l.pt')
    for image in files('medium_instance'):
        if cached(variant, 'medium_instance', image):
            continue
        start = time.monotonic()
        rgb = Image.open(image).convert('RGB')
        detections = models.detect(rgb, taxonomy('medium_instance')['classes'])
        runner.log(f'{variant}: {image.name}, {len(detections)} detections')
        raw = base64.b64encode(image.read_bytes()).decode()
        masks = []
        for detection in detections:
            mask = models.sam(raw, detection['bbox'])
            if mask.any():
                masks.append({**detection, 'points': runner.encode(mask), 'pixels': int(mask.sum())})
        save_prediction(variant, 'medium_instance', image, masks, time.monotonic()-start)
    del models
    gc.collect()


def predict_mask2former():
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    variant = M2F_VARIANT
    processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).eval()
    for task in TASKS:
        classes = set(taxonomy(task)['classes'])
        for image in files(task):
            if cached(variant, task, image):
                continue
            start = time.monotonic()
            rgb = Image.open(image).convert('RGB')
            runner.log(f'{variant}: {task}/{image.name}')
            with torch.inference_mode():
                outputs = model(**processor(images=rgb, return_tensors='pt'))
                prediction = processor.post_process_panoptic_segmentation(
                    outputs, threshold=.5, mask_threshold=.5, overlap_mask_area_threshold=.8,
                    # Transformers 4.57.6 can reuse IDs after fusing stuff. Leave IDs
                    # unique here; the lab's panoptic reader merges stuff by class.
                    label_ids_to_fuse=set(), target_sizes=[(rgb.height, rgb.width)])[0]
            segment_ids = [s['id'] for s in prediction['segments_info']]
            if len(segment_ids) != len(set(segment_ids)):
                raise RuntimeError('Duplicate panoptic segment IDs; refusing ambiguous masks')
            seg = prediction['segmentation'].numpy()
            masks = []
            for item in prediction['segments_info']:
                original = model.config.id2label[item['label_id']]
                label = ALIASES.get(original, original)
                if label not in classes:
                    continue
                mask = seg == item['id']
                if mask.any():
                    masks.append({'label': label, 'confidence': float(item['score']), 'points': runner.encode(mask), 'pixels': int(mask.sum())})
            save_prediction(variant, task, image, masks, time.monotonic()-start)
            del outputs, prediction
            gc.collect()
    del model
    gc.collect()


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1-x0)*max(0, y1-y0)
    return inter/max(1e-6, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def tiled_instances(models, image, targets):
    rgb = Image.open(image).convert('RGB')
    w, h = rgb.size
    cw, ch = int(w*.65), int(h*.65)
    windows = [(0,0,w,h)] + [(x,y,x+cw,y+ch) for y in (0,h-ch) for x in (0,w-cw)]
    detections = []
    for x0,y0,x1,y1 in windows:
        output = models.yolo.predict(rgb.crop((x0,y0,x1,y1)), imgsz=1280, conf=.35, iou=.5, device='cpu', verbose=False)[0]
        for box in output.boxes:
            label = output.names[int(box.cls.item())]
            if label in targets:
                a,b,c,d = box.xyxy[0].tolist()
                detections.append({'label': label, 'confidence': float(box.conf.item()), 'bbox': [a+x0,b+y0,c+x0,d+y0]})
    kept = []
    for det in sorted(detections, key=lambda d:-d['confidence']):
        if any(det['label']==old['label'] and box_iou(det['bbox'],old['bbox'])>.5 for old in kept):
            continue
        kept.append(det)
    groups = {}
    for det in kept:
        box = det['bbox']; window = windows[0]
        if min(box[2]-box[0], box[3]-box[1]) < 40:
            choices = [win for win in windows[1:] if win[0]<=box[0] and win[1]<=box[1] and win[2]>=box[2] and win[3]>=box[3]]
            if choices:
                window = max(choices, key=lambda win:min(box[0]-win[0],box[1]-win[1],win[2]-box[2],win[3]-box[3]))
        groups.setdefault(window, []).append(det)
    runner.log(f'tiled {image.name}: {len(kept)} detections; {len(groups)} SAM contexts')
    predictions = []
    for window, group in groups.items():
        buf = io.BytesIO(); rgb.crop(window).save(buf, format='PNG')
        raw = base64.b64encode(buf.getvalue()).decode()
        x0,y0,x1,y1 = window
        for det in group:
            a,b,c,d = det['bbox']
            crop_mask = models.sam(raw, [a-x0,b-y0,c-x0,d-y0])
            mask = np.zeros((h,w), bool);mask[y0:y1,x0:x1] = crop_mask
            if mask.any():
                predictions.append({**det,'points':runner.encode(mask),'pixels':int(mask.sum())})
    # A second guard removes duplicate masks caused by boxes from different crops.
    unique = []
    for pred in sorted(predictions, key=lambda m:-m['confidence']):
        mask = runner.decode(pred['points'])
        duplicate = False
        for other in unique:
            if other['label'] != pred['label']:
                continue
            om = runner.decode(other['points'])
            if (mask & om).sum()/max(1,(mask | om).sum()) > .85:
                duplicate = True;break
        if not duplicate:
            unique.append(pred)
    return unique


def predict_tiled():
    from ultralytics import YOLO
    models = runner.Models()
    models.yolo = YOLO('/cache/models/yolo11l.pt')
    for task in TASKS:
        tax = taxonomy(task);stuff = set(tax.get('stuff', []))
        for image in files(task):
            started = time.monotonic()
            temp_variant = 'tiled-things'
            if cached(temp_variant, task, image):
                things = json.loads(result_path(temp_variant,task,image).read_text())['masks']
            else:
                things = tiled_instances(models,image,set(tax['classes'])-stuff)
                save_prediction(temp_variant,task,image,things,time.monotonic()-started)
            if task == 'medium_instance':
                if not cached('yolo11l-tiled-sam',task,image):
                    save_prediction('yolo11l-tiled-sam',task,image,things,time.monotonic()-started)
                continue
            rgb = Image.open(image).convert('RGB')
            for variant in ('tiled-segformer','tiled-mask2former'):
                if cached(variant, task, image):
                    continue
                occupied = np.zeros((rgb.height,rgb.width),bool)
                masks = []
                for obj in things:
                    mask = runner.decode(obj['points']) & ~occupied
                    if mask.any():
                        occupied |= mask
                        masks.append({**obj,'points':runner.encode(mask),'pixels':int(mask.sum())})
                if variant == 'tiled-segformer':
                    seg, confidence, names = models.segment(rgb)
                    regions = [(label, seg==label_id, float(confidence[seg==label_id].mean()))
                               for label_id,label in names.items() if label in stuff and (seg==label_id).any()]
                else:
                    source = json.loads(result_path(M2F_VARIANT,task,image).read_text())['masks']
                    regions = [(m['label'],runner.decode(m['points']),m['confidence']) for m in source if m['label'] in stuff]
                for label,mask,confidence in regions:
                    mask &= ~occupied
                    if mask.any():
                        occupied |= mask
                        masks.append({'label':label,'confidence':confidence,'points':runner.encode(mask),'pixels':int(mask.sum())})
                save_prediction(variant,task,image,masks,time.monotonic()-started)
    del models
    gc.collect()


def predict_ensemble():
    """Consensus and mask-NMS use predictions only, not reference annotations."""
    for task in TASKS:
        tax = taxonomy(task); stuff = set(tax.get('stuff', []))
        for image in files(task):
            baseline = json.loads((ROOT/'.lab3-auto/predictions'/task/(image.stem+'.json')).read_text())
            source_masks = [('yolo11n', baseline['masks'])]
            for variant in ('yolo11l-sam', 'tiled-things', M2F_VARIANT):
                path = result_path(variant,task,image)
                if path.exists():
                    source_masks.append((variant,json.loads(path.read_text())['masks']))
            clusters = []
            for source, masks in source_masks:
                for m in masks:
                    if m['label'] in stuff:
                        continue
                    mask = runner.decode(m['points'])
                    item = {'source':source,'prediction':m,'mask':mask}
                    best, best_iou = None, .5
                    for group in clusters:
                        if group[0]['prediction']['label'] != m['label']:
                            continue
                        old = group[0]['mask']
                        iou = float((mask&old).sum()/max(1,(mask|old).sum()))
                        if iou > best_iou:
                            best, best_iou = group, iou
                    if best is None:
                        clusters.append([item])
                    else:
                        best.append(item)
            chosen = []
            for group in clusters:
                def agreement(candidate):
                    mask = candidate['mask']
                    values = [float((mask&other['mask']).sum()/max(1,(mask|other['mask']).sum())) for other in group]
                    return sum(values)/len(values)+.05*candidate['prediction']['confidence']
                selected = max(group,key=agreement)
                supports = len(set(item['source'] for item in group))
                confidence = max(item['prediction']['confidence'] for item in group)
                chosen.append({**selected['prediction'],'confidence':confidence,'support':supports})
            for variant, threshold, fill in [('ensemble-050',.5,False),('ensemble-065',.65,False),('ensemble-stufffill-050',.5,True)]:
                if fill and task != 'hard_panoptic':
                    continue
                if cached(variant,task,image):
                    continue
                started = time.monotonic()
                ordered = sorted((m for m in chosen if m['support']>=2 or m['confidence']>=threshold),
                                 key=lambda m: (m['support'], m['confidence']),reverse=True)
                masks = []; accepted = []
                occupied = np.zeros((baseline['height'],baseline['width']),bool)
                for m in ordered:
                    mask = runner.decode(m['points'])
                    if any(float((mask&other).sum()/max(1,(mask|other).sum()))>.85 for other in accepted):
                        continue
                    accepted.append(mask.copy())
                    if stuff:
                        mask &= ~occupied
                    if mask.any():
                        occupied |= mask
                        masks.append({**m,'points':runner.encode(mask),'pixels':int(mask.sum())})
                if stuff:
                    primary = baseline['masks'] + json.loads(result_path('tiled-segformer',task,image).read_text())['masks']
                    for label in tax['stuff']:
                        same = [runner.decode(m['points']) for m in primary if m['label']==label]
                        if not same:
                            continue
                        mask = np.logical_or.reduce(same) & ~occupied
                        if mask.any():
                            occupied |= mask
                            masks.append({'label':label,'confidence':1.,'points':runner.encode(mask),'pixels':int(mask.sum())})
                    if fill:
                        for m in json.loads(result_path(M2F_VARIANT,task,image).read_text())['masks']:
                            if m['label'] not in stuff:
                                continue
                            mask = runner.decode(m['points']) & ~occupied
                            if mask.any():
                                occupied |= mask
                                masks.append({**m,'points':runner.encode(mask),'pixels':int(mask.sum())})
                save_prediction(variant,task,image,masks,time.monotonic()-started)


def candidate_coco(variant, task):
    classes = taxonomy(task)['classes']
    coco = {'images': [], 'categories': [{'id': i+1, 'name': c} for i, c in enumerate(classes)], 'annotations': []}
    for i, image in enumerate(files(task), 1):
        pred = json.loads(result_path(variant, task, image).read_text())
        coco['images'].append({'id': i, 'file_name': image.name, 'height': pred['height'], 'width': pred['width']})
        for m in pred['masks']:
            mask = runner.decode(m['points'])
            rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
            bbox = coco_mask.toBbox(rle).tolist()
            rle['counts'] = rle['counts'].decode()
            coco['annotations'].append({'id': len(coco['annotations'])+1, 'image_id': i,
                'category_id': classes.index(m['label'])+1, 'segmentation': rle,
                'area': int(mask.sum()), 'bbox': bbox, 'iscrowd': 0})
    runner.write_json(WORK/'variants'/variant/task/'coco.json', coco)
    return coco


def evaluate(task, coco):
    # Evaluation only: references are not passed to predictors or mask generation.
    base = ROOT/'data'/REGISTRY[task]['path']
    tax = taxonomy(task)
    if task == 'medium_instance':
        result = metrics.score_instance(coco, json.loads((base/'groundtruth/instances.json').read_text()), set(tax['classes']))
        points = metrics.metric_to_points(result['value'], 32)
    else:
        pred = metrics.build_pred_panoptic(coco, set(tax['classes']), set(tax['stuff']))
        gt = metrics.load_panoptic_gt(base/'groundtruth/panoptic.json', base/'groundtruth/png',
                                    {int(k): v for k, v in tax['catid2name'].items()}, set(tax['classes']))
        result = metrics.score_panoptic(pred, gt, set(tax['classes']))
        points = metrics.metric_to_points(result['value'], 30, cap=metrics.PQ_CAP, floor=metrics.PQ_FLOOR)
    return {'points': points, 'result': result}


def evaluate_all():
    results = {}
    for task in TASKS:
        baseline = metrics._find_coco_json(metrics._read_members(WORK/'baseline'/(task+'.zip')))
        results[task] = {'baseline': evaluate(task, baseline)}
        variants = VARIANTS[task]
        for variant in variants:
            if all(result_path(variant, task, p).exists() for p in files(task)):
                results[task][variant] = evaluate(task, candidate_coco(variant, task))
        runner.log(f'EVALUATION {task}: ' + str({k: v['points'] for k,v in results[task].items()}))
    runner.write_json(WORK/'evaluation.json', results)
    return results


def promote(results):
    official_path = ROOT/'.lab3-auto/state.json'
    official = json.loads(official_path.read_text())
    promotion_path = WORK/'promotion.json'
    state = json.loads(promotion_path.read_text()) if promotion_path.exists() else {'tasks': {}}
    cvat = runner.CVAT()
    easy_before = {'zip': runner.sha(ROOT/'submissions/easy_semantic.zip'),
                   'annotations': runner.fingerprint(cvat.call('/api/tasks/3/annotations'))}
    def save():
        runner.write_json(promotion_path, state)
    for task, variants in results.items():
        best = max(variants, key=lambda v: variants[v]['result']['value'])
        if best == 'baseline':
            runner.log(task + ': keep baseline; no tested improvement')
            continue
        current_coco = metrics._find_coco_json(metrics._read_members(ROOT/'submissions'/(task+'.zip')))
        current = evaluate(task, current_coco)
        if variants[best]['result']['value'] <= current['result']['value'] + 1e-8:
            runner.log(task + ': current ZIP already as good or better; no changes')
            continue
        row = state['tasks'].setdefault(task, {'task_prefix': 'improved_'})
        if row.get('prediction_model') not in (None, best):
            row = state['tasks'][task] = {'task_prefix': 'improved_' + best + '_'}
        row['prediction_model'] = best
        row['prediction_override_dir'] = str((WORK/'variants'/best/task).relative_to(ROOT))
        save()
        previous = (ROOT/'submissions'/(task+'.zip')).read_bytes()
        try:
            runner.process(task, REGISTRY[task], cvat, None, state, save)
            exported = metrics._find_coco_json(metrics._read_members(ROOT/'submissions'/(task+'.zip')))
            actual = evaluate(task, exported)
            if actual['result']['value'] <= current['result']['value']:
                raise RuntimeError('Export did not improve the actual score; restoring previous ZIP')
        except Exception:
            (ROOT/'submissions'/(task+'.zip')).write_bytes(previous)
            raise
        row['score_before'] = current
        row['score_after'] = actual
        official['tasks'][task] = row.copy()
        runner.write_json(official_path, official)
        save()
        for image in files(task):
            src = WORK/'variants'/best/'overlays'/task/(image.stem+'.png')
            shutil.copy2(src, ROOT/'automation_previews'/task/src.name)
        runner.log(f"PROMOTED {task}: {current['points']} → {actual['points']}")
    after = {'zip': runner.sha(ROOT/'submissions/easy_semantic.zip'),
             'annotations': runner.fingerprint(cvat.call('/api/tasks/3/annotations'))}
    assert after == easy_before, 'Easy changed unexpectedly'
    official['easy_unchanged'] = True
    runner.write_json(official_path, official)
    runner.report(official)
    lines = ['# Cải thiện điểm Lab 3', '', 'Dự đoán được tạo từ ảnh bằng mô hình pretrained; reference đã phát chỉ dùng để đánh giá và chọn bản tốt hơn.',
             'Không thay scorer, không sao chép mask reference vào bài. Đây là điểm tự đánh giá sau đối chiếu reference, không phải đánh giá trên tập độc lập mới.', '',
             '| Task | Phương án | Điểm |', '|---|---|---:|']
    for task, variants in results.items():
        for variant, measured in variants.items():
            lines.append(f"| {task} | {variant} | {measured['points']} |")
    lines += ['', '## Phiên bản đã xuất lại từ CVAT', '']
    for task, row in state['tasks'].items():
        lines += [f"- {task}: {row.get('prediction_model')}; task {row.get('task_id')}, job {row.get('job_id')}; {row.get('annotation_count')} masks.",
                  f"  Điểm ZIP CVAT: {row.get('score_before',{}).get('points')} → {row.get('score_after',{}).get('points')}; SHA-256 `{row.get('zip_sha256')}`."]
    lines += ['', '- Easy giữ nguyên. Các task/checkpoint khác không được thay đổi.',
              '- Đã tắt fuse stuff trong Transformers 4.57.6 để tránh ID segment bị dùng lại; scorer gộp stuff theo class sau đó.',
              '- ZIP cũ và trạng thái cũ: `.lab3-improve/baseline/`. Cache dự đoán và đánh giá: `.lab3-improve/`.',
              '- Chất lượng vẫn cần kiểm trực quan; không xác nhận học viên đã kiểm hoặc tự vẽ thủ công.', '',
              'Nguồn mô hình: [YOLO11](https://docs.ultralytics.com/models/yolo11/), [Mask2Former COCO panoptic](https://huggingface.co/facebook/mask2former-swin-small-coco-panoptic).']
    (ROOT/'IMPROVEMENT_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--promote', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    (WORK/'baseline').mkdir(parents=True, exist_ok=True)
    for task in ('easy_semantic', *TASKS):
        dest = WORK/'baseline'/(task+'.zip')
        if not dest.exists():
            shutil.copy2(ROOT/'submissions'/(task+'.zip'), dest)
    if not (WORK/'baseline/state.json').exists():
        shutil.copy2(ROOT/'.lab3-auto/state.json', WORK/'baseline/state.json')
    if not args.evaluate_only:
        predict_yolo()
        predict_mask2former()
        predict_tiled()
        predict_ensemble()
    results = evaluate_all()
    if args.promote:
        promote(results)


if __name__ == '__main__':
    main()
