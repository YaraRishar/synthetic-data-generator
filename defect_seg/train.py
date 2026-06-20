"""Единая точка входа обучения (заменяет дублирующие model.py и saved_model.py).

Режимы:
  sweep  --- перебор доли синтетики 0.0..1.0 (по умолчанию), для исследования
           влияния синтетики на качество;
  single --- один прогон с заданными real-size/synthetic-size.

Результаты сохраняются в JSON (раньше только печатались в stdout и терялись).
Запуск:
  python train.py REAL SYNTH [EPOCHS BATCH TEST] [--mode single --real-size 0.5 ...]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping,
                                        ModelCheckpoint)

from defect_seg import data as seg_data
from defect_seg import model as seg_model
from defect_seg import viz as seg_viz
from defect_seg.config import TrainConfig, parse_args


def _make_callbacks(weights_path=""):
    # свежие callbacks на каждый fit (у них есть внутреннее состояние)
    cbs = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    if weights_path:
        Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
        cbs.append(ModelCheckpoint(filepath=weights_path, save_weights_only=True, verbose=1))
    return cbs


def _build_compiled(cfg: TrainConfig):
    model = seg_model.build_segmentation_model()
    model.compile(optimizer=Adam(learning_rate=cfg.learning_rate),
                  loss=seg_model.dice_loss,
                  metrics=seg_model.make_metrics())
    return model


def _evaluate_named(model, X_test, y_test, batch):
    values = model.evaluate(X_test, y_test, batch_size=batch, verbose=0)
    if not isinstance(values, (list, tuple)):
        values = [values]
    return dict(zip(model.metrics_names, [float(v) for v in values]))


def _fractions(cfg: TrainConfig):
    if cfg.mode == "single":
        return [(cfg.real_size, cfg.synthetic_size)]
    return [(round(1 - i / 10, 1), round(i / 10, 1)) for i in range(11)]


def run(cfg: TrainConfig, keras_callbacks=None, on_fraction=None) -> dict:
    """keras_callbacks --- доп. колбэки Keras на каждый fit (например, прогресс UI).
    on_fraction(idx, total, real_size, synth_size, metrics) --- после каждой доли."""
    rng = np.random.default_rng(cfg.seed)
    time_start = time.time()
    print("Загрузка датасетов...")
    X_real_full, y_real_full = seg_data.load_dataset(cfg.real)
    X_synth_full, y_synth_full = seg_data.load_dataset(cfg.synthetic)
    if cfg.max_images and cfg.max_images > 0:
        X_real_full, y_real_full = X_real_full[:cfg.max_images], y_real_full[:cfg.max_images]
        X_synth_full, y_synth_full = X_synth_full[:cfg.max_images], y_synth_full[:cfg.max_images]
    print(f"реальных: {len(X_real_full)}, синтетики: {len(X_synth_full)}")

    results = {"iou": [], "loss": [], "real_size": [], "synthetic_size": [],
               "real_elements": [], "synthetic_elements": [], "total_time": -1}

    fractions = _fractions(cfg)
    for frac_idx, (real_size, synthetic_size) in enumerate(fractions):
        tf.keras.backend.clear_session()  # не копить графы между прогонами
        real_params = (X_real_full, y_real_full, real_size)
        synth_params = (X_synth_full, y_synth_full, synthetic_size)

        X_mixed, y_mixed, X_test, y_test = seg_data.get_mixed_data(
            real_params, synth_params, test_size=cfg.test_size, rng=rng)
        real_n, synth_n = seg_data.get_number_of_elements(real_params, synth_params)

        cbs = _make_callbacks(cfg.weights)
        if keras_callbacks:
            cbs = cbs + list(keras_callbacks)
        model = _build_compiled(cfg)
        model.fit(X_mixed, y_mixed, validation_data=(X_test, y_test),
                  epochs=cfg.epochs, batch_size=cfg.batch, verbose=1,
                  callbacks=cbs)

        out_dir = Path(cfg.real) / "predictions" / f"pred{real_size}"
        sample = sorted(os.listdir(os.path.join(cfg.real, "images")))[::10]
        seg_viz.visualize_predictions(cfg.real, model, sample, out_dir,
                                      threshold=cfg.threshold)

        metrics = _evaluate_named(model, X_test, y_test, cfg.batch)
        iou = round(metrics.get("defect_iou", metrics.get("mean_iou", 0.0)), 4)
        loss = round(metrics.get("loss", 0.0), 4)
        results["iou"].append(iou)
        results["loss"].append(loss)
        results["real_size"].append(real_size)
        results["synthetic_size"].append(synthetic_size)
        results["real_elements"].append(real_n)
        results["synthetic_elements"].append(synth_n)
        print(f"real={real_size} synth={synthetic_size} -> iou={iou} loss={loss}")
        if on_fraction:
            on_fraction(frac_idx, len(fractions), real_size, synthetic_size,
                        {"iou": iou, "loss": loss})

    results["total_time"] = round(time.time() - time_start)

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"run_{int(time_start)}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Результаты сохранены: {out_file}")
    return results


def main(argv=None):
    cfg = parse_args(argv)
    run(cfg)


if __name__ == "__main__":
    # обратная совместимость с docker_runner: model.py REAL SYNTH EPOCHS BATCH TEST
    main(sys.argv[1:])
