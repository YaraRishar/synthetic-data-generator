"""Конфигурация обучения: вынесена из кода (раньше пути/гиперпараметры были
зашиты в model.py / saved_model.py или спрашивались через input())."""
import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    real: str
    synthetic: str
    epochs: int = 8
    batch: int = 8
    test_size: int = 60
    mode: str = "sweep"            # sweep | single
    real_size: float = 1.0        # для mode=single: доля реальных
    synthetic_size: float = 0.0   # для mode=single: доля синтетики
    threshold: float = 0.7
    learning_rate: float = 1e-3
    results_dir: str = "results"
    weights: str = ""             # путь к чекпойнту для загрузки/сохранения
    seed: int = 0
    max_images: int = 0           # 0 = без ограничения; иначе обрезать датасеты (ускорение/демо)


def parse_args(argv=None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Обучение сегментации дефектов")
    p.add_argument("real", help="путь к датасету реальных данных (images/ + bitmaps/)")
    p.add_argument("synthetic", help="путь к датасету синтетики")
    p.add_argument("epochs", nargs="?", type=int, default=8)
    p.add_argument("batch", nargs="?", type=int, default=8)
    p.add_argument("test_size", nargs="?", type=int, default=60)
    p.add_argument("--mode", choices=["sweep", "single"], default="sweep")
    p.add_argument("--real-size", type=float, default=1.0)
    p.add_argument("--synthetic-size", type=float, default=0.0)
    p.add_argument("--threshold", type=float, default=0.7)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--weights", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-images", type=int, default=0)
    a = p.parse_args(argv)
    return TrainConfig(
        real=a.real, synthetic=a.synthetic, epochs=a.epochs, batch=a.batch,
        test_size=a.test_size, mode=a.mode, real_size=a.real_size,
        synthetic_size=a.synthetic_size, threshold=a.threshold,
        learning_rate=a.learning_rate, results_dir=a.results_dir,
        weights=a.weights, seed=a.seed, max_images=a.max_images)
