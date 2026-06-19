"""График IoU/Loss в зависимости от доли синтетики.

Читает результаты из results/*.json (их пишет train.py). Если файлов нет —
падает на встроенные данные прежних прогонов. Можно указать файл аргументом:
    python plotter.py [results/run_XXX.json] [--save out.png]
"""
import argparse
import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt


def _embedded():
    iou = [[0.901, 0.9023, 0.897, 0.8963, 0.8841, 0.8676, 0.8767, 0.7176, 0.8311, 0.7954, 0.7769],
           [0.905, 0.9103, 0.8796, 0.8752, 0.8602, 0.8503, 0.8528, 0.8373, 0.8209, 0.7926, 0.7491],
           [0.8991, 0.9028, 0.8949, 0.8971, 0.8725, 0.8877, 0.8742, 0.8499, 0.8597, 0.803, 0.7521]]
    loss = [[0.141, 0.132, 0.1452, 0.1417, 0.1573, 0.1794, 0.1651, 0.3953, 0.2329, 0.2811, 0.2985],
            [0.1394, 0.1243, 0.1665, 0.1648, 0.1845, 0.2029, 0.193, 0.2214, 0.2452, 0.2734, 0.345],
            [0.1425, 0.137, 0.1473, 0.1457, 0.1743, 0.1597, 0.1744, 0.1998, 0.193, 0.2719, 0.346]]

    def mid(arr):
        return [round(sum(col) / len(col), 4) for col in zip(*arr)]

    return {"iou": mid(iou), "loss": mid(loss),
            "synthetic_size": [round(i / 10, 1) for i in range(11)]}


def load_results(path=None):
    if path is None:
        files = sorted(glob.glob(os.path.join("results", "*.json")))
        path = files[-1] if files else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    print("results/*.json не найдены — использую встроенные данные")
    return _embedded()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="?", default=None)
    p.add_argument("--save", default=None, help="сохранить в файл вместо показа")
    a = p.parse_args()
    if a.save:
        matplotlib.use("Agg")

    data = load_results(a.results)
    x = data["synthetic_size"]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Доля синтетических данных")
    ax1.set_ylabel("Метрика IoU", color="blue")
    ax1.plot(x, data["iou"], color="blue", marker="o", label="IoU")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss", color="red")
    ax2.plot(x, data["loss"], color="red", marker="x", label="Loss")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0, 1)

    ax1.grid()
    plt.title("Дополнение синтетическими данными")
    if a.save:
        plt.savefig(a.save, dpi=120, bbox_inches="tight")
        print(f"Сохранено: {a.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
