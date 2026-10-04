"""
Trains three CNN backbones on the PCOS ultrasound dataset, compares them on
accuracy / precision / recall / F1 / specificity (+ AUC), picks the best one
and saves it as CNN_MODEL_PATH so the rest of MAPCOS (CNNClassificationAgent,
Grad-CAM, orchestrator) can use it unchanged.

Selection rule: the model with the highest F1 on the VALIDATION split wins
(ties broken by AUC, then log loss). Corrupt images are skipped automatically.
"""

import gc
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mapcos.config import BATCH_SIZE, CNN_MODEL_PATH, DATA_DIR_VISION, IMG_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_VAR = [0.229 ** 2, 0.224 ** 2, 0.225 ** 2]

CANDIDATES = {
    "EfficientNetB0": (EfficientNetB0, lambda: layers.Rescaling(255.0)),
    "DenseNet121": (DenseNet121,
                    lambda: layers.Normalization(mean=IMAGENET_MEAN, variance=IMAGENET_VAR)),
    "MobileNetV2": (MobileNetV2, lambda: layers.Rescaling(2.0, offset=-1.0)),
}

CLASSES = ["notinfected", "infected"]
OUT_DIR = "cnn_comparison"


def make_clean_copy(data_dir, out_root=None):
    out_root = out_root or os.path.join(tempfile.gettempdir(), "pcos_clean")
    shutil.rmtree(out_root, ignore_errors=True)
    bad, good = [], 0
    for split in ("train", "test"):
        for cls in CLASSES:
            src_dir = os.path.join(data_dir, split, cls)
            dst_dir = os.path.join(out_root, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for fname in sorted(os.listdir(src_dir)):
                src = os.path.join(src_dir, fname)
                try:
                    with Image.open(src) as im:
                        im.load()
                except Exception:
                    bad.append(src)
                    continue
                dst = os.path.join(dst_dir, fname)
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                good += 1
    print(f"Clean dataset: {good} readable images, {len(bad)} skipped")
    for b in bad:
        print("  skipped unreadable file:", b)
    return out_root


def build_data_generators(data_dir):
    train_dir, test_dir = f"{data_dir}/train", f"{data_dir}/test"
    common = dict(target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
                  class_mode="binary", classes=CLASSES, color_mode="rgb")

    aug = ImageDataGenerator(rescale=1. / 255, rotation_range=15, zoom_range=0.1,
                             horizontal_flip=True, validation_split=0.15)
    plain = ImageDataGenerator(rescale=1. / 255, validation_split=0.15)
    test_plain = ImageDataGenerator(rescale=1. / 255)

    train_gen = aug.flow_from_directory(train_dir, subset="training", seed=42, **common)
    val_gen = plain.flow_from_directory(train_dir, subset="validation", shuffle=False, **common)
    test_gen = test_plain.flow_from_directory(test_dir, shuffle=False, **common)
    return train_gen, val_gen, test_gen


def build_model(name, weights="imagenet"):
    backbone_cls, preprocess = CANDIDATES[name]
    base = backbone_cls(include_top=False, weights=weights,
                        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocess()(inp)
    x = base(x)
    backbone = models.Model(inp, x, name="backbone")

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs, name=name), base


def _compile(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])


def _callbacks():
    return [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                             restore_best_weights=True)]


def train_one(name, train_gen, val_gen, class_weight, weights, head_epochs, finetune_epochs):
    model, base = build_model(name, weights)
    _compile(model, 1e-4)
    model.fit(train_gen, validation_data=val_gen, epochs=head_epochs,
              class_weight=class_weight, callbacks=_callbacks(), verbose=2)

    if finetune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        _compile(model, 1e-5)
        model.fit(train_gen, validation_data=val_gen, epochs=finetune_epochs,
                  class_weight=class_weight, callbacks=_callbacks(), verbose=2)
    return model


def evaluate(model, gen, threshold=0.5):
    gen.reset()
    probs = model.predict(gen, verbose=0).ravel()
    y_true = gen.classes.astype(int)
    y_pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "auc": roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else float("nan"),
        "log_loss": log_loss(y_true, np.clip(probs, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_comparison(test_df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["accuracy", "precision", "recall", "f1", "specificity"]
    ax = test_df[metrics].plot(kind="bar", figsize=(9, 4.5), ylim=(0, 1), rot=0)
    ax.set_title("CNN comparison on the test set")
    ax.set_ylabel("score")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(model_names=None, data_dir=DATA_DIR_VISION, weights="imagenet",
         head_epochs=15, finetune_epochs=10, out_dir=OUT_DIR, clean_data=True):
    model_names = model_names or list(CANDIDATES)
    os.makedirs(out_dir, exist_ok=True)
    if clean_data:
        data_dir = make_clean_copy(data_dir)

    train_gen, val_gen, test_gen = build_data_generators(data_dir)
    print("Class indices:", train_gen.class_indices, "(1 = infected = positive class)")

    classes = np.unique(train_gen.classes)
    cw = compute_class_weight("balanced", classes=classes, y=train_gen.classes)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print("Class weights:", class_weight)

    val_rows, test_rows = {}, {}
    for name in model_names:
        print(f"\n{'=' * 20} Training {name} {'=' * 20}")
        model = train_one(name, train_gen, val_gen, class_weight, weights,
                          head_epochs, finetune_epochs)
        model.save(os.path.join(out_dir, f"{name}.h5"))
        val_rows[name] = evaluate(model, val_gen)
        test_rows[name] = evaluate(model, test_gen)
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    val_df = pd.DataFrame(val_rows).T
    test_df = pd.DataFrame(test_rows).T
    shown = ["accuracy", "precision", "recall", "f1", "specificity", "auc", "log_loss"]

    pd.set_option("display.width", 200)
    print("\nVALIDATION metrics (used to pick the winner)")
    print(val_df[shown].round(4))
    print("\nTEST metrics")
    print(test_df[shown].round(4))
    print("\nTEST confusion counts (positive = infected)")
    print(test_df[["tn", "fp", "fn", "tp"]].astype(int))

    ranked = val_df.assign(neg_log_loss=-val_df["log_loss"]).sort_values(
        ["f1", "auc", "neg_log_loss"], ascending=False)
    best = ranked.index[0]
    print(f"\n>>> Best CNN (highest validation F1, then AUC, then log loss): {best}")

    val_df.to_csv(os.path.join(out_dir, "validation_metrics.csv"))
    test_df.to_csv(os.path.join(out_dir, "test_metrics.csv"))
    plot_comparison(test_df, os.path.join(out_dir, "test_comparison.png"))
    with open(os.path.join(out_dir, "best_cnn.json"), "w") as f:
        json.dump({"best_model": best, "selected_by": "validation F1",
                   "test_metrics": {k: float(v) for k, v in test_df.loc[best].items()}}, f, indent=2)

    shutil.copy(os.path.join(out_dir, f"{best}.h5"), CNN_MODEL_PATH)
    print(f"Best model copied to {CNN_MODEL_PATH}")
    return best, val_df, test_df


if __name__ == "__main__":
    main()
