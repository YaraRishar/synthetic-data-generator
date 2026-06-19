"""УСТАРЕЛО: одиночный прогон с чекпойнтом теперь делается через train.py:

    python train.py REAL SYNTH --mode single --real-size 1.0 --weights ckpt.weights.h5

Без аргументов запускается single-режим на example_datasets_scratches/.
"""
import sys

from config import TrainConfig
from train import main, run

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        run(TrainConfig(
            real="example_datasets_scratches/real",
            synthetic="example_datasets_scratches/synthetic",
            mode="single", real_size=1.0, synthetic_size=0.0,
            test_size=60, epochs=8, batch=8, threshold=0.8,
            learning_rate=1e-4,
            weights="example_datasets_scratches/checkpoint.weights.h5"))
