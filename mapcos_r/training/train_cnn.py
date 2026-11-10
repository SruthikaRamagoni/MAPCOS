"""
Trains the EfficientNetB0 backbone used by CNNClassificationAgent.
Run this once, in a Kaggle Notebook with a GPU turned on, to produce
cnn_agent_best.h5, then point CNNClassificationAgent at that file.

Run: python -m mapcos.training.train_cnn
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mapcos.config import IMG_SIZE, BATCH_SIZE, DATA_DIR_VISION, CNN_MODEL_PATH

TRAIN_DIR = f"{DATA_DIR_VISION}/train"
TEST_DIR = f"{DATA_DIR_VISION}/test"


def build_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255, rotation_range=15, zoom_range=0.1,
        horizontal_flip=True, validation_split=0.15
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="binary", subset="training", color_mode="rgb"
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="binary", subset="validation", color_mode="rgb"
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False, color_mode="rgb"
    )
    return train_gen, val_gen, test_gen


def build_cnn_model():
    base = EfficientNetB0(include_top=False, weights="imagenet",
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model, base


def main():
    train_gen, val_gen, _ = build_data_generators()
    model, base = build_cnn_model()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        CNN_MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True
    )

    # Phase 1: train the classification head only
    model.fit(train_gen, validation_data=val_gen, epochs=15,
              callbacks=[early_stop, checkpoint])

    # Phase 2: unfreeze the top of the backbone, fine-tune at a low LR
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    model.fit(train_gen, validation_data=val_gen, epochs=10,
              callbacks=[early_stop, checkpoint])

    print(f"Best model saved to {CNN_MODEL_PATH}")


if __name__ == "__main__":
    main()
