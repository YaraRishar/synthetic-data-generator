"""Визуализация предсказаний: наложение истинной (зелёной) и предсказанной
(красной) масок на изображение. Предсказания считаются батчем."""
from pathlib import Path

import cv2 as cv
import numpy as np

from defect_seg.cv_io import imread_unicode, imwrite_unicode

IMAGE_SIZE = (200, 200)


def _overlay(img_gray, true_mask, pred_mask):
    h, w = img_gray.shape[:2]
    img_color = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    true_c = np.zeros((h, w, 3), dtype=np.uint8)
    pred_c = np.zeros((h, w, 3), dtype=np.uint8)
    true_c[true_mask > 0] = [0, 255, 0]
    pred_c[pred_mask > 0] = [0, 0, 255]
    overlay = img_color.copy()
    tm = true_mask > 0
    pm = pred_mask > 0
    overlay[tm] = np.clip(overlay[tm] * 0.3 + true_c[tm] * 0.7, 0, 255).astype(np.uint8)
    overlay[pm] = np.clip(overlay[pm] * 0.2 + pred_c[pm] * 0.8, 0, 255).astype(np.uint8)
    return overlay


def visualize_predictions(dataset_path, model, image_names, out_dir, threshold=0.5,
                          image_size=IMAGE_SIZE):
    """Сохранить overlay для списка изображений (батч-инференс).

    image_names - итерируемое имён файлов из dataset_path/images.
    out_dir создаётся при необходимости (кроссплатформенно).
    """
    dataset_path = Path(dataset_path)
    image_dir = dataset_path / "images"
    mask_dir = dataset_path / "bitmaps"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_names = list(image_names)
    originals, true_masks, batch, kept = [], [], [], []
    for name in image_names:
        img = imread_unicode(str(image_dir / name), cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]
        tm = imread_unicode(str(mask_dir / name.replace("image", "bitmap")), cv.IMREAD_GRAYSCALE)
        if tm is None:
            tm = np.zeros((h, w), dtype=np.uint8)
        tm = cv.resize(tm, (w, h))
        _, tm = cv.threshold(tm, 127, 255, cv.THRESH_BINARY)

        inp = cv.resize(img, image_size).astype(np.float32) / 255.0
        originals.append(img)
        true_masks.append(tm)
        batch.append(np.expand_dims(inp, axis=-1))
        kept.append(name)

    if not batch:
        return 0

    preds = model.predict(np.stack(batch), verbose=0)
    written = 0
    for img, tm, pred, name in zip(originals, true_masks, preds, kept):
        h, w = img.shape[:2]
        pm = (pred > threshold).astype(np.uint8) * 255
        pm = cv.resize(pm[..., 0] if pm.ndim == 3 else pm, (w, h))
        _, pm = cv.threshold(pm, 127, 255, cv.THRESH_BINARY)
        overlay = _overlay(img, tm, pm)
        imwrite_unicode(str(out_dir / f"{Path(name).stem}_overlay.jpg"), overlay)
        written += 1
    return written
