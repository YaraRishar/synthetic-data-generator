import os
import cv2 as cv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, Reshape)
from tensorflow.keras.optimizers import Adam


def load_dataset(dataset_path, image_size=(200, 200)):
    images = []
    annotations = []

    image_dir = os.path.join(dataset_path, "images")
    markup_dir = os.path.join(dataset_path, "csvs")

    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, image_size)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)

        base_name = os.path.splitext(img_file)[0]
        csv_path = os.path.join(markup_dir, f"{base_name}.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, sep=";")
            boxes = df[["X", "Y", "H", "W"]].values
            boxes = boxes / [image_size[0], image_size[1], image_size[0], image_size[1]]
            images.append(image)
            annotations.append(boxes)

    return np.array(images), np.array(annotations, dtype=object)


def build_detection_model(input_shape=(200, 200, 1), max_boxes=15):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (2, 2), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)

    outputs = Dense(max_boxes * 4, activation='sigmoid', name='boxes')(x)
    outputs = Reshape((max_boxes, 4))(outputs)

    return Model(inputs=inputs, outputs=outputs)


def prepare_labels(y_annotations, max_boxes=15):
    box_coords = []
    for boxes in y_annotations:
        padded = np.zeros((max_boxes, 4))
        n = min(len(boxes), max_boxes)
        padded[:n] = boxes[:n]
        box_coords.append(padded)
    return np.array(box_coords)


X, y = load_dataset("example_datasets_scratches/synthetic")
y = prepare_labels(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = build_detection_model()
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='mse',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    batch_size=10,
                    verbose=1)


def visualize_prediction(image_path, model, threshold=0.0001):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    h, w = img.shape[:2]

    img_input = cv.resize(img, (200, 200))
    img_input = img_input.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_input, axis=(0, -1))

    preds = model.predict(img_input)[0]
    for box in preds:
        if np.max(box) > threshold:
            x, y, bh, bw = box
            x1, y1 = int(x * w), int(y * h)
            x2, y2 = int((x + bw) * w), int((y + bh) * h)
            cv.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 1)

    print(cv.imwrite(f"{image_path.split('.')[0]}_resulting.jpg", img_color))


visualize_prediction("example_datasets_scratches/synthetic/images/image6.jpg", model)
visualize_prediction("example_datasets_scratches/synthetic/images/image83.jpg", model)
