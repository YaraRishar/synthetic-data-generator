"""Слой обратной совместимости.

Реализация переехала в seg_data / seg_model / seg_viz, чтобы убрать дублирование
(раньше одни и те же функции жили в utils.py, model.py и saved_model.py).
Старые импорты `import utils; utils.load_dataset(...)` продолжают работать.
"""
from pathlib import Path

from seg_data import load_dataset, get_mixed_data, get_number_of_elements
from seg_model import dice_coefficient, dice_loss, build_segmentation_model
from seg_viz import visualize_predictions as _visualize_predictions

__all__ = [
    "load_dataset", "get_mixed_data", "get_number_of_elements",
    "dice_coefficient", "dice_loss", "build_segmentation_model",
    "visualize_predictions",
]


def visualize_predictions(real_dataset_path, model, image_name, folder, threshold=0.01):
    """Совместимая обёртка над seg_viz (старый вызов — по одному изображению)."""
    out_dir = Path(real_dataset_path) / "predictions" / folder
    return _visualize_predictions(real_dataset_path, model, [image_name], out_dir,
                                  threshold=threshold)
