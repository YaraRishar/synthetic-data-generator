"""Архитектура сегментации и метрики/потери.

Единственное место, где описана U-Net-модель и dice-loss --- раньше этот код был
скопирован в model.py, saved_model.py и utils.py.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                                      Concatenate, Dropout)
from tensorflow.keras.regularizers import l2

IMAGE_SIZE = (200, 200)
INPUT_SHAPE = (200, 200, 1)


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def make_metrics():
    """Метрики сегментации.

    Defect-IoU (target_class_ids=[1]) --- это IoU именно по дефекту, а не по фону
    (раньше saved_model.py ошибочно мерил класс 0 = фон). mean-IoU оставлен для
    сравнения с прежними прогонами model.py.
    """
    return [
        tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name="defect_iou"),
        tf.keras.metrics.BinaryIoU(target_class_ids=(0, 1), threshold=0.5, name="mean_iou"),
    ]


def build_segmentation_model(input_shape=INPUT_SHAPE, weight_decay=1e-4):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same",
                   kernel_regularizer=l2(weight_decay))(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same",
                   kernel_regularizer=l2(weight_decay))(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same",
                   kernel_regularizer=l2(weight_decay))(pool2)
    conv3 = Dropout(0.3)(conv3)
    up1 = UpSampling2D((2, 2))(conv3)
    concat1 = Concatenate()([up1, conv2])
    conv4 = Conv2D(64, (3, 3), activation="relu", padding="same",
                   kernel_regularizer=l2(weight_decay))(concat1)
    up2 = UpSampling2D((2, 2))(conv4)
    concat2 = Concatenate()([up2, conv1])
    conv5 = Conv2D(32, (3, 3), activation="relu", padding="same",
                   kernel_regularizer=l2(weight_decay))(concat2)
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv5)
    return Model(inputs=inputs, outputs=outputs)
