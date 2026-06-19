"""УСТАРЕЛО: логика обучения переехала в train.py (+ seg_model/seg_data/seg_viz).

Файл оставлен как точка входа для docker_runner, который запускает
`python /tf/model.py REAL SYNTH EPOCHS BATCH TEST`. Просто делегирует в train.
"""
import sys

from train import main

if __name__ == "__main__":
    main(sys.argv[1:])
