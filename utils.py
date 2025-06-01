import os

import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


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
    cv.imwrite(f"{real_dataset_path}/predictions/{folder}/{filename}_overlay.jpg", overlay)


def get_number_of_elements(real_params: tuple, synthetic_params: tuple):
    X_real_full, _, real_size = real_params
    X_synthetic_full, _, synthetic_size = synthetic_params
    real_elements_num = int(len(X_real_full) * real_size)
    synthetic_elements_num = int(len(X_synthetic_full) * synthetic_size)
    return real_elements_num, synthetic_elements_num


def get_mixed_data(real_params: tuple, synthetic_params: tuple):
    X_real_full, y_real_full, real_size = real_params
    X_synthetic_full, y_synthetic_full, synthetic_size = synthetic_params

    real_elements_num, synthetic_elements_num = get_number_of_elements(real_params, synthetic_params)

    real_index_list = np.random.choice(len(X_real_full), real_elements_num, replace=False)
    synthetic_index_list = np.random.choice(len(X_synthetic_full), synthetic_elements_num, replace=False)

    X_real, y_real = X_real_full[real_index_list], y_real_full[real_index_list]
    X_synthetic, y_synthetic = X_synthetic_full[synthetic_index_list], y_synthetic_full[synthetic_index_list]

    X_mixed = np.concatenate((X_real, X_synthetic))
    y_mixed = np.concatenate((y_real, y_synthetic))

    return X_mixed, y_mixed


# def get_mixed_data2(real_data: tuple, synthetic_data: tuple, test_size=60, real_ratio=0.7):
#     X_real, y_real = real_data
#     X_synth, y_synth = synthetic_data
#
#     X_real_train, X_test, y_real_train, y_test = train_test_split(
#         X_real, y_real,
#         test_size=test_size,
#         random_state=1)
#
#     n_real_train = len(X_real_train)
#     n_synth_train = int((n_real_train / real_ratio) * (1 - real_ratio))
#
#     synth_indices = np.random.choice(len(X_synth), n_synth_train, replace=False)
#     X_synth_train = X_synth[synth_indices]
#     y_synth_train = y_synth[synth_indices]
#
#     X_train_mixed = np.concatenate((X_real_train, X_synth_train))
#     y_train_mixed = np.concatenate((y_real_train, y_synth_train))
#
#     shuffle_idx = np.random.permutation(len(X_train_mixed))
#     X_train_mixed = X_train_mixed[shuffle_idx]
#     y_train_mixed = y_train_mixed[shuffle_idx]
#
#     return (X_train_mixed, y_train_mixed), (X_test, y_test)
#
#
# def prepare_datasets_with_paths(real_data: tuple, synthetic_data: tuple, real_paths: list, test_size=60,
#                                 real_ratio=0.7):
#     """
#     Args:
#         real_data: (X_real_full, y_real_full)
#         synthetic_data: (X_synthetic_full, y_synthetic_full)
#         real_paths: list of paths corresponding to X_real_full
#         test_size: number of test samples
#         real_ratio: mixing ratio
#     """
#     X_real, y_real = real_data
#     X_synth, y_synth = synthetic_data
#
#     # Split real data with paths
#     X_real_train, X_test, y_real_train, y_test, _, test_paths = train_test_split(
#         X_real, y_real, real_paths,
#         test_size=test_size,
#         random_state=42
#     )
#
#     n_real_train = len(X_real_train)
#     n_synth_train = int((n_real_train / real_ratio) * (1 - real_ratio))
#
#     # Select synthetic samples
#     synth_indices = np.random.choice(len(X_synth), n_synth_train, replace=False)
#     X_synth_train = X_synth[synth_indices]
#     y_synth_train = y_synth[synth_indices]
#
#     X_train = np.concatenate((X_real_train, X_synth_train))
#     y_train = np.concatenate((y_real_train, y_synth_train))
#
#     shuffle_idx = np.random.permutation(len(X_train))
#     X_train = X_train[shuffle_idx]
#     y_train = y_train[shuffle_idx]
#
#     return X_train, y_train, X_test, y_test, test_paths