"""Тесты synthetic-data-generator (пакет defect_seg).

Запуск без pytest:   python tests/test_segmentation.py
Запуск с pytest:     pytest tests/
"""
import os
import sys
import tempfile

import numpy as np

# работаем из корня проекта (относительные пути к example_datasets_scratches)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defect_seg import data as seg_data
from defect_seg import model as seg_model
from defect_seg import viz as seg_viz
from defect_seg import contours

EX = "example_datasets_scratches"


def test_no_leakage_between_train_and_test():
    # каждая строка уникальна по значению -> легко проверить пересечение
    n = 50
    X = np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1)
    y = X.copy()
    rng = np.random.default_rng(0)
    Xmix, ymix, Xtest, ytest = seg_data.get_mixed_data(
        (X, y, 0.5), (np.empty((0, 1, 1, 1)), np.empty((0, 1, 1, 1)), 0.0),
        test_size=10, rng=rng)
    train_ids = set(Xmix.ravel().tolist())
    test_ids = set(Xtest.ravel().tolist())
    assert train_ids.isdisjoint(test_ids), "тест пересекается с трейном (утечка)"
    assert len(test_ids) == 10
    assert len(train_ids) == int(n * 0.5)


def test_get_mixed_data_raises_when_too_many():
    n = 20
    X = np.zeros((n, 1, 1, 1), dtype=np.float32)
    try:
        seg_data.get_mixed_data((X, X, 1.0), (X, X, 0.0), test_size=10)
    except ValueError:
        return
    raise AssertionError("ожидался ValueError при test_size + train > n")


def test_mix_combines_real_and_synthetic():
    nr, ns = 40, 30
    Xr = np.zeros((nr, 1, 1, 1), dtype=np.float32)
    Xs = np.ones((ns, 1, 1, 1), dtype=np.float32)
    Xmix, ymix, Xtest, _ = seg_data.get_mixed_data(
        (Xr, Xr, 0.5), (Xs, Xs, 0.5), test_size=10, rng=np.random.default_rng(1))
    # 0.5*40 реальных (из пула 30) + 0.5*30 синтетики = 20 + 15
    assert len(Xmix) == 20 + 15
    assert len(Xtest) == 10


def test_dice_loss_bounds():
    import tensorflow as tf
    yt = tf.ones((1, 4, 4, 1))
    assert abs(float(seg_model.dice_loss(yt, yt))) < 1e-3          # совпадение -> ~0
    yp = tf.zeros((1, 4, 4, 1))
    assert float(seg_model.dice_loss(yt, yp)) > 0.9               # промах -> ~1


def test_model_output_shape():
    m = seg_model.build_segmentation_model()
    out = m.predict(np.zeros((2, 200, 200, 1), dtype=np.float32), verbose=0)
    assert out.shape == (2, 200, 200, 1)
    assert 0.0 <= out.min() and out.max() <= 1.0                  # sigmoid


def test_load_dataset_real_example():
    if not os.path.isdir(os.path.join(EX, "real", "images")):
        print("  (пропуск: нет example_datasets_scratches)")
        return
    X, y = seg_data.load_dataset(os.path.join(EX, "real"))
    assert len(X) == len(y) and len(X) > 0
    assert X.shape[1:] == (200, 200, 1)
    assert set(np.unique(y)).issubset({0.0, 1.0})                 # бинарные маски


def test_contours_csv_format():
    bmp = None
    for root, _, files in os.walk("output_example"):
        for f in files:
            if f.endswith(".jpg"):
                bmp = os.path.join(root, f)
                break
        if bmp:
            break
    if bmp is None:
        print("  (пропуск: нет bitmap для контуров)")
        return
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "contours0_x.csv")
        contours.contours_csv(idx=0, image_path=bmp, path_to_csv=out)
        with open(out, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        assert lines[0].startswith("Contour #")
        if len(lines) > 1:
            first = lines[1].split(";")
            assert first[0] == "0"                                # номер контура в начале


def test_visualize_predictions_writes_overlay():
    if not os.path.isdir(os.path.join(EX, "real", "images")):
        print("  (пропуск: нет example датасета)")
        return
    names = sorted(os.listdir(os.path.join(EX, "real", "images")))[:2]
    m = seg_model.build_segmentation_model()
    with tempfile.TemporaryDirectory() as d:
        written = seg_viz.visualize_predictions(
            os.path.join(EX, "real"), m, names, d, threshold=0.5)
        assert written == len(names)
        assert len(os.listdir(d)) == len(names)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} прошли")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
