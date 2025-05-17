import os
import time

import cv2 as cv
import keras
import numpy as np
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


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
        # print(f"loading image {img_file}")
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


def build_segmentation_model(input_shape=(200, 200, 1), weight_decay=1e-4):
    inputs = Input(shape=input_shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay))(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay))(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay))(pool2)

    conv3 = Dropout(0.3)(conv3)

    up1 = UpSampling2D((2, 2))(conv3)
    concat1 = Concatenate()([up1, conv2])

    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay))(concat1)

    up2 = UpSampling2D((2, 2))(conv4)
    concat2 = Concatenate()([up2, conv1])

    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_regularizer=l2(weight_decay))(concat2)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    return Model(inputs=inputs, outputs=outputs)


def visualize_predictions(image_name, threshold=0.01):
    image_path = os.path.join(real_test_dataset_path, "images/")
    mask_path = os.path.join(real_test_dataset_path, "bitmaps/")

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
    cv.imwrite(f"{real_test_dataset_path}/pred{float(real_size)}/{filename}_overlay.jpg", overlay)


checkpoint_path = "example_datasets_scratches/checkpoint.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

synthetic_dataset_path = "example_datasets_scratches/synthetic/"
real_dataset_path = "example_datasets_scratches/real/"
real_test_dataset_path = "example_datasets_scratches/real_test/"

X_synthetic_full, y_synthetic_full = load_dataset(synthetic_dataset_path)
X_real_full, y_real_full = load_dataset(real_dataset_path)
X_test, y_test = load_dataset(real_test_dataset_path)

real_size = 1
synthetic_size = 1 - real_size

real_elements_num = int(len(X_real_full) * real_size)
synthetic_elements_num = int(len(X_synthetic_full) * synthetic_size)

real_index_list = np.random.choice(len(X_real_full), real_elements_num, replace=False)
synthetic_index_list = np.random.choice(len(X_synthetic_full), synthetic_elements_num, replace=False)

X_real, y_real = X_real_full[real_index_list], y_real_full[real_index_list]
X_synthetic, y_synthetic = X_synthetic_full[synthetic_index_list], y_synthetic_full[synthetic_index_list]

X_mixed = np.concatenate((X_real, X_synthetic))
y_mixed = np.concatenate((y_real, y_synthetic))

model = build_segmentation_model()
model.compile(optimizer=Adam(learning_rate=1e-4),
              # loss=tf.keras.losses.MeanSquaredLogarithmicError(
              #     reduction='sum_over_batch_size',
              #     name='mean_squared_logarithmic_error'),
              # loss=keras.losses.MeanSquaredError(
              #     reduction="sum_over_batch_size", name="mean_squared_error", dtype=None),
              loss=dice_loss,
              # loss=keras.losses.BinaryFocalCrossentropy(
              #       apply_class_balancing=False,
              #       alpha=0.25,
              #       gamma=2.0,
              #       from_logits=False,
              #       label_smoothing=0.0,
              #       axis=-1,
              #       reduction="sum_over_batch_size",
              #       name="binary_focal_crossentropy",
              #       dtype=None),
              # loss=keras.losses.CosineSimilarity(
              #     axis=-1, reduction="sum_over_batch_size", name="cosine_similarity", dtype=None),
              metrics=[tf.keras.metrics.IoU(num_classes=2,
                                            target_class_ids=[0],
                                            name=None,
                                            dtype=None,
                                            ignore_class=None,
                                            sparse_y_true=True,
                                            sparse_y_pred=True,
                                            axis=-1), 'accuracy'])
model.summary()

do_load = input("Load model? (y/N)\n")
time_start = time.time()
if do_load.lower() == "y":
    model.load_weights(checkpoint_path)
else:
    history = model.fit(X_mixed, y_mixed,
                        validation_data=(X_test, y_test),
                        epochs=8,
                        batch_size=8,
                        verbose=1,
                        callbacks=[cp_callback, reduce_lr, early_stopping])

time_end = time.time()
for image_file in os.listdir(os.path.join(real_test_dataset_path, "images")):
    visualize_predictions(image_file, threshold=0.8)

loss, iou, acc = model.evaluate(X_test, y_test, batch_size=8)
print(f"iou: {iou:.4f} | acc: {acc:.4f} | loss: {loss:.4f} || "
         f"real size: {real_size} | synthetic_size: {synthetic_size}")
print(f"real_num {real_elements_num} | synthetic_num {synthetic_elements_num}")
print(f"time total: {time_end - time_start}")
