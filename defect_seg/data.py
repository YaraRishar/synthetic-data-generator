"""Загрузка датасетов и формирование смешанных выборок (реал + синтетика).

Главное исправление: тестовая выборка отделяется от обучающей ДО набора трейна,
поэтому одни и те же реальные изображения больше не попадают и в train, и в test
(раньше get_mixed_data делал два независимых np.random.choice -> утечка).
"""
import os
from pathlib import Path

import cv2 as cv
import numpy as np

from defect_seg.cv_io import imread_unicode

IMAGE_SIZE = (200, 200)


def _mask_path_for(image_file: str, mask_dir: str) -> str:
    # imageN.jpg -> bitmapN.jpg; меняем только имя файла, не путь
    base_name = Path(image_file).stem
    return os.path.join(mask_dir, f"{base_name.replace('image', 'bitmap')}.jpg")


def load_dataset(dataset_path, image_size=IMAGE_SIZE):
    """Загрузить пары (изображение, маска) из dataset_path/images и /bitmaps."""
    images, masks = [], []
    image_dir = os.path.join(dataset_path, "images")
    mask_dir = os.path.join(dataset_path, "bitmaps")
    for img_file in sorted(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_file)
        image = imread_unicode(img_path, cv.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv.resize(image, image_size).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)

        mask_path = _mask_path_for(img_file, mask_dir)
        if not os.path.exists(mask_path):
            continue
        mask = imread_unicode(mask_path, cv.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = cv.resize(mask, image_size)
        _, mask = cv.threshold(mask, 127, 1, cv.THRESH_BINARY)
        mask = np.expand_dims(mask.astype(np.float32), axis=-1)

        images.append(image)
        masks.append(mask)

    return np.array(images), np.array(masks)


def get_number_of_elements(real_params: tuple, synthetic_params: tuple):
    X_real_full, _, real_size = real_params
    X_synthetic_full, _, synthetic_size = synthetic_params
    real_elements_num = int(len(X_real_full) * real_size)
    synthetic_elements_num = int(len(X_synthetic_full) * synthetic_size)
    return real_elements_num, synthetic_elements_num


def get_mixed_data(real_params: tuple, synthetic_params: tuple, test_size=60, rng=None):
    """Собрать обучающую смесь и НЕпересекающийся тест из реальных данных.

    real_params/synthetic_params = (X_full, y_full, fraction).
    """
    rng = np.random.default_rng() if rng is None else rng
    X_real_full, y_real_full, _ = real_params
    X_synthetic_full, y_synthetic_full, _ = synthetic_params

    real_elements_num, synthetic_elements_num = get_number_of_elements(
        real_params, synthetic_params)

    n_real = len(X_real_full)
    if test_size + real_elements_num > n_real:
        raise ValueError(
            f"test_size ({test_size}) + реальных в трейне ({real_elements_num}) "
            f"> всего реальных ({n_real}); уменьшите test_size или долю реальных.")

    # сначала отделяем тест, трейн берём ТОЛЬКО из оставшихся индексов
    perm = rng.permutation(n_real)
    test_idx, pool_idx = perm[:test_size], perm[test_size:]
    X_test, y_test = X_real_full[test_idx], y_real_full[test_idx]

    real_idx = rng.choice(pool_idx, real_elements_num, replace=False)
    X_real, y_real = X_real_full[real_idx], y_real_full[real_idx]

    n_synth = len(X_synthetic_full)
    synth_idx = rng.choice(n_synth, min(synthetic_elements_num, n_synth), replace=False)
    X_synth, y_synth = X_synthetic_full[synth_idx], y_synthetic_full[synth_idx]

    if len(X_real) and len(X_synth):
        X_mixed = np.concatenate((X_real, X_synth))
        y_mixed = np.concatenate((y_real, y_synth))
    elif len(X_real):
        X_mixed, y_mixed = X_real, y_real
    else:
        X_mixed, y_mixed = X_synth, y_synth

    return X_mixed, y_mixed, X_test, y_test
