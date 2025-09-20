import os
import re

import cv2 as cv
import numpy as np
import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def load_dataset(dataset_path, image_size=(200, 200)):
    images = []
    masks = []
    image_dir = os.path.join(dataset_path, "images")
    mask_dir = os.path.join(dataset_path, "bitmaps")
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, image_size)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)

        base_name = os.path.splitext(img_file)[0]
        mask_path = os.path.join(mask_dir, f"{base_name.replace('image', 'bitmap')}.jpg")

        if os.path.exists(mask_path):
            mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
            mask = cv.resize(mask, image_size)
            _, mask = cv.threshold(mask, 127, 1, cv.THRESH_BINARY)
            mask = mask.astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            images.append(image)
            masks.append(mask)

    return np.array(images), np.array(masks)


def visualize_predictions(real_dataset_path, model, image_name, folder, threshold=0.01):
    index = re.search(r'image(\d+)', image_name).group(1)
    image_name = "image" + index + ".jpg"
    image_path = os.path.join(real_dataset_path, "images/")
    mask_path = os.path.join(real_dataset_path, "bitmaps/")

    img = cv.imread(image_path + image_name, cv.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    true_mask = cv.imread(mask_path + image_name.replace("image", "bitmap"), cv.IMREAD_GRAYSCALE)
    true_mask = cv.resize(true_mask, (w, h))
    _, true_mask = cv.threshold(true_mask, 127, 255, cv.THRESH_BINARY)
    img_input = cv.resize(img, (200, 200))
    img_input = img_input.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=(0, -1))
    pred_mask = model.predict(img_input)[0]
    pred_mask = (pred_mask > threshold).astype(np.uint8) * 255
    pred_mask = cv.resize(pred_mask, (w, h))
    _, pred_mask = cv.threshold(pred_mask, 127, 255, cv.THRESH_BINARY)
    true_mask_color = np.zeros((h, w, 3), dtype=np.uint8)
    pred_mask_color = np.zeros((h, w, 3), dtype=np.uint8)

    true_mask_color[true_mask > 0] = [0, 255, 0]
    pred_mask_color[pred_mask > 0] = [0, 0, 255]

    overlay = img_color.copy()
    overlay[true_mask > 0] = overlay[true_mask > 0] * 0.3 + true_mask_color[true_mask > 0] * 0.7
    overlay[pred_mask > 0] = overlay[pred_mask > 0] * 0.2 + pred_mask_color[pred_mask > 0] * 0.8

    filename = os.path.splitext(os.path.basename(image_name))[0]
    filepath = f"{real_dataset_path}predictions/{folder}/{filename}_overlay.jpg"
    cv.imwrite(filepath, overlay)


def get_number_of_elements(real_params: tuple, synthetic_params: tuple):
    X_real_full, _, real_size = real_params
    X_synthetic_full, _, synthetic_size = synthetic_params
    real_elements_num = int(len(X_real_full) * real_size)
    synthetic_elements_num = int(len(X_synthetic_full) * synthetic_size)
    return real_elements_num, synthetic_elements_num


def get_mixed_data(real_params: tuple, synthetic_params: tuple, test_size=60):
    X_real_full, y_real_full, real_size = real_params
    X_synthetic_full, y_synthetic_full, synthetic_size = synthetic_params

    real_elements_num, synthetic_elements_num = get_number_of_elements(real_params, synthetic_params)

    test_index_list = np.random.choice(len(X_real_full), test_size, replace=False)
    X_test = X_real_full[test_index_list]
    y_test = y_real_full[test_index_list]

    real_index_list = np.random.choice(len(X_real_full), real_elements_num, replace=False)
    X_real, y_real = X_real_full[real_index_list], y_real_full[real_index_list]

    synthetic_index_list = np.random.choice(len(X_synthetic_full), synthetic_elements_num, replace=False)
    X_synthetic, y_synthetic = X_synthetic_full[synthetic_index_list], y_synthetic_full[synthetic_index_list]

    X_mixed = np.concatenate((X_real, X_synthetic))
    y_mixed = np.concatenate((y_real, y_synthetic))

    return X_mixed, y_mixed, X_test, y_test
