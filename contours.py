import csv

import cv2 as cv

from cv_io import imread_unicode


def get_contours(image_path: str) -> list:
    """ Найти контуры дефекта на bitmap, аппроксимировать кривые найденных контуров """

    image = imread_unicode(image_path)  # кириллические имена дефектов -> Unicode-safe
    if image is None:
        raise FileNotFoundError(f"не удалось прочитать {image_path}")
    im_grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, temp = cv.threshold(im_grayscale, 125, 200, 0)
    temp = cv.dilate(temp, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv.findContours(temp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    new_contours = []
    for i in range(len(contours)):
        # создать новый контур с меньшим кол-вом точек (лучше чем 0.006 * длина кривой контура)
        approximated_contours = cv.approxPolyDP(contours[i], 0.006 * cv.arcLength(contours[i], True), True)
        new_contours.append(approximated_contours)

    return new_contours


def get_bound_box(image_path: str) -> list:
    contours = get_contours(image_path)
    return [tuple(cv.boundingRect(contour)) for contour in contours]


def contours_csv(idx: int, image_path: str, path_to_csv: str):
    """ Сформировать csv файл контуров.
    idx — номер изображения в датасете (используется только в имени файла выше).
    Формат строки: <номер_контура>; x1; y1; x2; y2; ... """

    contours = get_contours(image_path)
    with open(path_to_csv, mode="w", newline="") as file:
        csv_writer = csv.writer(file, delimiter=";")
        csv_writer.writerow(["Contour #", "Contour coords"])
        for contour_number, contour in enumerate(contours):
            contour_list = list(contour.ravel())
            contour_list.insert(0, contour_number)  # номер контура в начало строки
            csv_writer.writerow(contour_list)


def bound_box_csv(idx: int, image_path: str, path_to_csv: str):
    path_to_csv = path_to_csv.replace("contours", "bound_box")
    bb_list = get_bound_box(image_path)
    with open(path_to_csv, mode="w", newline="") as file:
        csv_writer = csv.writer(file, delimiter=";")
        csv_writer.writerow(["BB #", "X", "Y", "W", "H"])
        for i, bb in enumerate(bb_list):
            csv_writer.writerow([i, *bb])
