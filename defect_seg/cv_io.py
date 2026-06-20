"""Unicode-безопасный ввод/вывод изображений.

Чтение и запись идут через np.fromfile + cv.imdecode/imencode, что корректно
работает с путями, содержащими не-ASCII символы (например кириллические имена
дефектов), на любой ОС, включая Windows.
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
