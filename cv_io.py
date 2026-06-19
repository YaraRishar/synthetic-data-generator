"""Unicode-безопасный ввод/вывод изображений.

cv2.imread/imwrite на Windows используют ANSI-API и молча возвращают None /
ничего не пишут для путей с не-ASCII символами. А имена дефектов кириллические
(bitmap0_14_царапины.jpg), поэтому на Windows весь пайплайн контуров ломался.
Чтение/запись через np.fromfile + cv.imdecode/imencode работает на любой ОС.
"""
import os

import cv2 as cv
import numpy as np


def imread_unicode(path, flags=cv.IMREAD_COLOR):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv.imdecode(data, flags)


def imwrite_unicode(path, img) -> bool:
    path = str(path)
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return bool(ok)
