import os
import time

import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from keras.src.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

import utils


def build_segmentation_model(input_shape=(200, 200, 1), weight_decay=1e-4):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(weight_decay))(inputs)
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


time_start = time.time()
synthetic_dataset_path = "example_datasets_scratches/synthetic/"
real_dataset_path = "example_datasets_scratches/real/"

real_size = 1.0
results_dict = {"iou": [], "loss": [],
                "real_size": [], "synthetic_size": [],
                "real_elements": [], "synthetic_elements": [], "total_time": -1}
folder_names = []
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

X_synthetic_full, y_synthetic_full = utils.load_dataset(synthetic_dataset_path)
X_real_full, y_real_full = utils.load_dataset(real_dataset_path)

for i in range(11):
    real_size = round(1 - i/10, 1)
    synthetic_size = round(1 - real_size, 1)

    real_params = X_real_full, y_real_full, real_size
    synthetic_params = X_synthetic_full, y_synthetic_full, synthetic_size
    # synthetic_size = 1
    X_mixed, y_mixed = utils.get_mixed_data(real_params, synthetic_params)
    # X_mixed, y_mixed, X_test, y_test = utils.get_mixed_data2()

    real_elements_num, synthetic_elements_num = utils.get_number_of_elements(real_params, synthetic_params)

    model = build_segmentation_model()
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=utils.dice_loss,
                  metrics=[keras.metrics.BinaryIoU(target_class_ids=(0, 1), threshold=0.5, name=None, dtype=None)])
    history = model.fit(X_mixed, y_mixed,
                        validation_data=(X_test, y_test),
                        epochs=8,
                        batch_size=8,
                        verbose=1,
                        callbacks=[reduce_lr, early_stopping])

    folder_name = "pred" + str(real_size)
    folder_names.append(folder_name)
    for image_file in os.listdir(os.path.join(real_dataset_path, "images")):
        utils.visualize_predictions(real_dataset_path, model, image_file, folder_name, threshold=0.7)

    loss, iou = model.evaluate(X_test, y_test, batch_size=8)
    loss, iou = round(loss, 4), round(iou, 4)

    results_dict["iou"].append(iou)
    results_dict["loss"].append(loss)
    results_dict["real_size"].append(real_size)
    results_dict["synthetic_size"].append(synthetic_size)
    results_dict["synthetic_elements"].append(synthetic_elements_num)
    results_dict["real_elements"].append(real_elements_num)

time_end = time.time()
results_dict["total_time"] = round(time_end - time_start)
print(results_dict)
print(folder_names)