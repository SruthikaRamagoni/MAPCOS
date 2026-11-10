
"""
Trains three CNN backbones on the PCOS ultrasound dataset, compares them on
accuracy / precision / recall / F1 / specificity (+ AUC), picks the best one
and saves it as CNN_MODEL_PATH so the rest of MAPCOS (CNNClassificationAgent,
Grad-CAM, orchestrator) can use it unchanged.

Run in a Kaggle notebook with a GPU and Internet ON (the ImageNet weights are
downloaded on first use):

    !python compare_cnns_kaggle.py        (standalone)
    !python -m mapcos.training.compare_cnns   (inside the MAPCOS repo)

LEAKAGE FIX (why this version differs from the original)
--------------------------------------------------------
The Kaggle dataset ships with augmented copies of the same ultrasound scans,
so a normal folder/random split puts near-identical images in train AND test,
which gives ~100% scores. This version:

  1. Pools every image from the dataset's train/ and test/ folders.
  2. Computes a perceptual hash (pHash) of each image, also for its flipped
     versions, and groups images whose hashes are within HASH_THRESHOLD bits.
  3. Keeps ONE image per group (KEEP_ONE_PER_GROUP) and drops any group whose
     members carry conflicting labels.
  4. Splits train / val / test BY GROUP (stratified), so no scan, and no copy
     of it, can appear in two splits.
  5. Applies augmentation to the training generator only.
  6. Prints how many of the ORIGINAL test images had a near-duplicate in the
     original train folder, so the leak is measured, not guessed.

Selection rule: the model with the highest F1 on the VALIDATION split wins
(ties broken by AUC, then log loss). Test-set metrics are reported for all
three models, so the winner's test numbers are an unbiased estimate of its
performance. The split is saved to cnn_comparison/splits.csv for reproducibility.

Limitation: pHash catches duplicates, flips and brightness/contrast changes
well, but heavy rotations or crops can escape it. If scores are still ~1.0
after this fix, raise HASH_THRESHOLD or test on an external dataset.
"""

import gc
import json
import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scipy.fft import dctn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Inside the MAPCOS repo the project config is used. Standalone on Kaggle the
# defaults below apply (edit them if your config uses different values).
try:
    from mapcos.config import BATCH_SIZE, CNN_MODEL_PATH, DATA_DIR_VISION, IMG_SIZE
except ImportError:
    IMG_SIZE = 224
    BATCH_SIZE = 32
    CNN_MODEL_PATH = "cnn_agent_best.h5"
    _CANDIDATE_DIRS = [
        "/kaggle/input/datasets/anaghachoudhari/pcos-detection-using-ultrasound-images/data",
        "/kaggle/input/pcos-detection-using-ultrasound-images/data",
    ]
    DATA_DIR_VISION = next((p for p in _CANDIDATE_DIRS if os.path.isdir(p)), _CANDIDATE_DIRS[0])

# The generators feed images scaled to [0, 1] (same as CNNClassificationAgent).
# Each backbone below starts with the layer(s) that convert [0, 1] into the
# range that backbone's ImageNet weights expect. Putting this INSIDE the model
# keeps the agent and Grad-CAM code unchanged for whichever CNN wins.
#   EfficientNetB0 expects raw 0-255 (it normalises internally)
#   MobileNetV2    expects [-1, 1]
#   DenseNet121    expects ImageNet mean/std normalisation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_VAR = [0.229 ** 2, 0.224 ** 2, 0.225 ** 2]

CANDIDATES = {
    "EfficientNetB0": (EfficientNetB0, lambda: layers.Rescaling(255.0)),
    "DenseNet121": (DenseNet121,
                    lambda: layers.Normalization(mean=IMAGENET_MEAN, variance=IMAGENET_VAR)),
    "MobileNetV2": (MobileNetV2, lambda: layers.Rescaling(2.0, offset=-1.0)),
}

# Class order matters: index 1 must be the positive (PCOS) class so the sigmoid
# output is P(infected), which is what CNNClassificationAgent assumes.
CLASSES = ["notinfected", "infected"]
OUT_DIR = "cnn_comparison"

SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15            # fraction of the WHOLE dataset used for validation
HASH_THRESHOLD = 6         # max differing bits (of 64) to call two images "the same scan"
KEEP_ONE_PER_GROUP = True  # keep a single image per duplicate group


# --------------------------------------------------------------------------- #
# Leakage-safe data preparation
# --------------------------------------------------------------------------- #
def _phash_bits(gray32):
    """64-bit perceptual hash from a 32x32 grayscale array (low-frequency DCT)."""
    low = dctn(gray32, norm="ortho")[:8, :8]
    return (low > np.median(low)).ravel()


def collect_images(data_dir):
    """Pool the dataset's train/ and test/ folders into one table.

    Skips unreadable/corrupt images (Kaggle input is read-only, so they cannot
    be deleted) and computes a pHash for the image and its flipped variants.
    Returns (DataFrame[filepath, label, orig_split], hash bits [N, 4, 64]).
    """
    rows, bits, bad = [], [], []
    for split in ("train", "test"):
        for cls in CLASSES:
            src_dir = os.path.join(data_dir, split, cls)
            for fname in sorted(os.listdir(src_dir)):
                src = os.path.join(src_dir, fname)
                try:
                    with Image.open(src) as im:
                        im.load()
                        width, height = im.size
                        full_gray = im.convert("L")
                        mean_gray = float(np.asarray(full_gray).mean())
                        gray = np.asarray(full_gray.resize((32, 32), Image.LANCZOS),
                                          dtype=np.float32)
                except Exception:
                    bad.append(src)
                    continue
                variants = (gray, gray[:, ::-1], gray[::-1, :], gray[::-1, ::-1])
                bits.append([_phash_bits(np.ascontiguousarray(v)) for v in variants])
                rows.append({"filepath": src, "label": cls, "orig_split": split,
                             "width": width, "height": height,
                             "filesize": os.path.getsize(src), "mean_gray": mean_gray})
    print(f"Collected {len(rows)} readable images, {len(bad)} skipped")
    for b in bad:
        print("  skipped unreadable file:", b)
    return pd.DataFrame(rows), np.array(bits, dtype=bool)


def hamming_matrix(bits):
    """Pairwise Hamming distance between images, taking the minimum over the
    original / horizontally / vertically / doubly flipped hash of the second
    image, so flipped copies count as duplicates."""
    A = bits[:, 0].astype(np.float32)
    D = np.full((len(A), len(A)), 64, dtype=np.float32)
    for v in range(bits.shape[1]):
        B = bits[:, v].astype(np.float32)
        matches = A @ B.T + (1 - A) @ (1 - B).T
        D = np.minimum(D, 64 - matches)
    return D


def report_leakage(df, D, thr):
    """How many of the ORIGINAL test images have a near-duplicate in the ORIGINAL train folder."""
    tr = np.where(df["orig_split"] == "train")[0]
    te = np.where(df["orig_split"] == "test")[0]
    if len(tr) == 0 or len(te) == 0:
        return
    frac = float((D[np.ix_(te, tr)].min(axis=1) <= thr).mean())
    print(f"Leakage check: {frac:.1%} of the original test images have a near-duplicate "
          f"(<= {thr} bits) in the original train folder")


def threshold_sweep(D, thresholds=(2, 4, 6, 8, 10)):
    """Show how many groups / how large the biggest group is at several thresholds,
    to help choose HASH_THRESHOLD (want a big group count and a small largest group)."""
    print("Threshold sweep (bits -> groups, largest group):")
    for t in thresholds:
        n, lab = connected_components(csr_matrix(D <= t), directed=False)
        print(f"  <= {t:2d} bits: {n:5d} groups, largest = {np.bincount(lab).max()}")


def shortcut_check(df):
    """Can a model that never looks at the picture content (only image size, file
    size and average brightness) already separate the classes? If its AUC is far
    above 0.5, the classes differ by a shortcut (source / format / brightness) and
    high CNN scores can't be trusted."""
    from sklearn.ensemble import RandomForestClassifier
    feats = ["width", "height", "filesize", "mean_gray"]
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    ytr, yva = (tr["label"] == "infected").astype(int), (va["label"] == "infected").astype(int)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(tr[feats], ytr)
    auc = roc_auc_score(yva, rf.predict_proba(va[feats])[:, 1])
    print(f"Shortcut check (metadata only, no pixels): validation AUC = {auc:.3f}")
    if auc > 0.7:
        print("  WARNING: image size / file size / brightness alone predicts the label. "
              "The dataset has a shortcut; high CNN scores may not reflect real PCOS detection.")


def build_group_split(df, D, thr=HASH_THRESHOLD, keep_one=KEEP_ONE_PER_GROUP):
    """Group near-duplicates, then split train/val/test by group (stratified)."""
    n_groups, group = connected_components(csr_matrix(D <= thr), directed=False)
    df = df.assign(group=group)

    sizes = df.groupby("group").size()
    print(f"{len(df)} images -> {n_groups} groups at threshold {thr} "
          f"(largest group: {sizes.max()}, images in multi-image groups: "
          f"{int(sizes[sizes > 1].sum())})")
    if sizes.max() > 25:
        print("  WARNING: a very large group suggests the threshold is too loose; "
              "consider lowering HASH_THRESHOLD.")

    # A group whose copies carry different labels is label noise -> drop it
    mixed = df.groupby("group")["label"].nunique()
    mixed = mixed[mixed > 1].index
    if len(mixed):
        print(f"Dropping {len(mixed)} groups with conflicting labels")
        df = df[~df["group"].isin(mixed)]

    if keep_one:
        df = df.sort_values("filepath").drop_duplicates("group")
        print(f"Kept one image per group: {len(df)} images")

    g = df.groupby("group", as_index=False).agg(label=("label", "first"))
    g_trval, g_test = train_test_split(g, test_size=TEST_FRAC, stratify=g["label"],
                                       random_state=SEED)
    g_train, g_val = train_test_split(g_trval, test_size=VAL_FRAC / (1 - TEST_FRAC),
                                      stratify=g_trval["label"], random_state=SEED)
    split_of = {}
    for name, part in (("train", g_train), ("val", g_val), ("test", g_test)):
        split_of.update({int(k): name for k in part["group"]})
    df = df.assign(split=df["group"].map(split_of)).reset_index(drop=True)

    assert df.groupby("group")["split"].nunique().max() == 1, "a group spans two splits!"
    print(pd.crosstab(df["split"], df["label"]))
    return df


def build_data_generators(df):
    common = dict(x_col="filepath", y_col="label", target_size=(IMG_SIZE, IMG_SIZE),
                  batch_size=BATCH_SIZE, class_mode="binary", classes=CLASSES,
                  color_mode="rgb")

    # Augmentation is applied to the TRAINING generator only
    aug = ImageDataGenerator(rescale=1. / 255, rotation_range=15, zoom_range=0.1,
                             horizontal_flip=True)
    plain = ImageDataGenerator(rescale=1. / 255)

    part = lambda s: df[df["split"] == s].reset_index(drop=True)
    train_gen = aug.flow_from_dataframe(part("train"), shuffle=True, seed=SEED, **common)
    # No augmentation and no shuffling for val/test, so metrics are stable
    # and comparable between models.
    val_gen = plain.flow_from_dataframe(part("val"), shuffle=False, **common)
    test_gen = plain.flow_from_dataframe(part("test"), shuffle=False, **common)
    return train_gen, val_gen, test_gen


# --------------------------------------------------------------------------- #
# Models / training / evaluation (unchanged)
# --------------------------------------------------------------------------- #
def build_model(name, weights="imagenet"):
    backbone_cls, preprocess = CANDIDATES[name]
    base = backbone_cls(include_top=False, weights=weights,
                        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    # Wrapper named "backbone" = preprocessing + pretrained CNN. This is the
    # layer Grad-CAM looks up (LAST_CONV_LAYER in config.py).
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
    # New callback objects per phase so early-stopping state is not carried over.
    return [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                             restore_best_weights=True)]


def train_one(name, train_gen, val_gen, class_weight, weights, head_epochs, finetune_epochs):
    model, base = build_model(name, weights)
    # Phase 1: train the new classification head only
    _compile(model, 1e-4)
    model.fit(train_gen, validation_data=val_gen, epochs=head_epochs,
              class_weight=class_weight, callbacks=_callbacks(), verbose=2)

    # Phase 2: unfreeze the top 30 layers of the backbone, fine-tune at a low LR
    if finetune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        _compile(model, 1e-5)
        model.fit(train_gen, validation_data=val_gen, epochs=finetune_epochs,
                  class_weight=class_weight, callbacks=_callbacks(), verbose=2)
    return model


def evaluate(model, gen, threshold=0.5):
    """Metrics with 'infected' (PCOS) as the positive class."""
    gen.reset()
    probs = model.predict(gen, verbose=0).ravel()
    y_true = np.asarray(gen.classes, dtype=int)   # .classes is a plain list here
    y_pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),   # sensitivity
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
         head_epochs=15, finetune_epochs=10, out_dir=OUT_DIR,
         hash_threshold=HASH_THRESHOLD, keep_one_per_group=KEEP_ONE_PER_GROUP):
    model_names = model_names or list(CANDIDATES)
    os.makedirs(out_dir, exist_ok=True)

    # 1-4: pool, hash, group, split by group (no scan can span two splits)
    df, bits = collect_images(data_dir)
    D = hamming_matrix(bits)
    report_leakage(df, D, hash_threshold)
    threshold_sweep(D)
    df = build_group_split(df, D, hash_threshold, keep_one_per_group)
    shortcut_check(df)
    df[["filepath", "label", "orig_split", "group", "split"]].to_csv(
        os.path.join(out_dir, "splits.csv"), index=False)
    del D, bits

    # 5: augmentation on the training generator only
    train_gen, val_gen, test_gen = build_data_generators(df)
    print("Class indices:", train_gen.class_indices, "(1 = infected = positive class)")

    classes = np.unique(np.asarray(train_gen.classes))
    cw = compute_class_weight("balanced", classes=classes, y=np.asarray(train_gen.classes))
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print("Class weights:", class_weight)

    val_rows, test_rows = {}, {}
    for name in model_names:
        print(f"\n{'=' * 20} Training {name} {'=' * 20}")
        model = train_one(name, train_gen, val_gen, class_weight, weights,
                          head_epochs, finetune_epochs)
        model.save(os.path.join(out_dir, f"{name}.h5"))   # save first, evaluate after
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

    # Winner = best validation F1; ties broken by validation AUC, then lowest log loss
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
