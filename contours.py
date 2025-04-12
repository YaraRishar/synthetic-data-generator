import cv2 as cv
import csv
import numpy as np


def get_contours(image_path: str) -> list:
    """ Найти контуры дефекта на bitmap, аппроксимировать кривые найденных контуров """

    image = cv.imread(image_path)
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
    bb_list = []
    for contour in contours:
        bb_list.append(tuple(cv.boundingRect(contour)))

    return bb_list


def contours_csv(idx: int, image_path: str, path_to_csv: str):
    """ Сформировать csv файл контуров
    idx - номер изображения в датасете
    формат C x1 y1 x2 y2 ... xn-1 yn-1 xn yn"""

    contours = get_contours(image_path)
    with open(path_to_csv, mode='x', newline='') as file:
        csv_writer = csv.writer(file, delimiter=';')
        csv_writer.writerow(["Contour #", "Contour coords"])
        for contour in contours:
            contour_list = list(contour.ravel())
            contour_list.insert(idx, 0)
            csv_writer.writerow(contour_list)


def bound_box_csv(idx: int, image_path: str, path_to_csv: str):

    path_to_csv = path_to_csv.replace("contours", "bound_box")
    bb_list = get_bound_box(image_path)
    with open(path_to_csv, mode='x', newline='') as file:
        csv_writer = csv.writer(file, delimiter=';')
        csv_writer.writerow(["BB #", "X", "Y", "W", "H"])
        for i in range(len(bb_list)):
            bb = list(bb_list[i])
            bb.insert(0, i)
            csv_writer.writerow(bb)
