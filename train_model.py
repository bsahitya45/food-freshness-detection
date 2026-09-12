import os
import numpy as np
import tensorflow as tf
from PIL import Image

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = r"D:\archive\dataset"
MODEL_PATH = r"D:\archive\food_freshness_model.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 8
SEED = 42


# ============================================================
# START
# ============================================================

print("=" * 60)
print("FOOD FRESHNESS CLASSIFIER")
print("=" * 60)


# ============================================================
# DATASET FOLDERS
# ============================================================

fresh_dirs = [
    os.path.join(DATASET_DIR, "fresh_bread"),
    os.path.join(DATASET_DIR, "fresh_dairy"),
    os.path.join(DATASET_DIR, "fresh_fruits"),
    os.path.join(DATASET_DIR, "fresh_vegetables"),
]

spoiled_dirs = [
    os.path.join(DATASET_DIR, "spoiled_bread"),
    os.path.join(DATASET_DIR, "spoiled_dairy"),
    os.path.join(DATASET_DIR, "spoiled_fruits"),
    os.path.join(DATASET_DIR, "spoiled_vegetables"),
]


# ============================================================
# COLLECT IMAGES
# ============================================================

image_paths = []
labels = []

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# -------------------------
# FRESH = 0
# -------------------------

for folder in fresh_dirs:

    if not os.path.exists(folder):
        print("WARNING - folder not found:", folder)
        continue

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(extensions):

                image_paths.append(
                    os.path.join(root, filename)
                )

                labels.append(0)


# -------------------------
# SPOILED = 1
# -------------------------

for folder in spoiled_dirs:

    if not os.path.exists(folder):
        print("WARNING - folder not found:", folder)
        continue

    for root, dirs, files in os.walk(folder):

        for filename in files:

            if filename.lower().endswith(extensions):

                image_paths.append(
                    os.path.join(root, filename)
                )

                labels.append(1)


print()
print(
    "Images found before checking:",
    len(image_paths)
)


# ============================================================
# CHECK IMAGES
# ============================================================

print()
print("=" * 60)
print("CHECKING IMAGES")
print("=" * 60)
print()

good_paths = []
good_labels = []

bad_images = []


for i in range(len(image_paths)):

    path = image_paths[i]
    label = labels[i]

    try:

        with Image.open(path) as img:

            # Actually load the image
            img.load()

            # Make sure it can be converted to RGB
            img.convert("RGB")

        good_paths.append(path)
        good_labels.append(label)

    except Exception as e:

        bad_images.append(path)

        print(
            "Skipping bad image:",
            path
        )

    if (i + 1) % 250 == 0:

        print(
            f"Checked {i + 1}/{len(image_paths)} images..."
        )


# ============================================================
# CLEAN DATA
# ============================================================

image_paths = np.array(good_paths)

labels = np.array(
    good_labels,
    dtype=np.float32
)


print()
print("=" * 60)
print("IMAGE CHECK COMPLETE")
print("=" * 60)

print(
    "Valid images:",
    len(image_paths)
)

print(
    "Bad images skipped:",
    len(bad_images)
)

print(
    "Fresh images:",
    int(np.sum(labels == 0))
)

print(
    "Spoiled images:",
    int(np.sum(labels == 1))
)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(image_paths) == 0:

    raise RuntimeError(
        "No valid images were found."
    )


if np.sum(labels == 0) == 0:

    raise RuntimeError(
        "No Fresh images were found."
    )


if np.sum(labels == 1) == 0:

    raise RuntimeError(
        "No Spoiled images were found."
    )


# ============================================================
# SHUFFLE DATA
# ============================================================

rng = np.random.default_rng(SEED)

indices = rng.permutation(
    len(image_paths)
)

image_paths = image_paths[indices]

labels = labels[indices]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split = int(
    len(image_paths) * 0.80
)

train_paths = image_paths[:split]

train_labels = labels[:split]

val_paths = image_paths[split:]

val_labels = labels[split:]


print()
print("=" * 60)
print("DATASET SPLIT")
print("=" * 60)

print(
    "Training images:",
    len(train_paths)
)

print(
    "Validation images:",
    len(val_paths)
)


# ============================================================
# PIL IMAGE LOADER
# ============================================================

def load_image_with_pil(path):

    # Convert TensorFlow bytes to normal Python string
    path = path.decode("utf-8")

    try:

        with Image.open(path) as img:

            # Convert every image to RGB
            img = img.convert("RGB")

            # Resize to MobileNetV2 input size
            img = img.resize(
                IMG_SIZE,
                Image.Resampling.LANCZOS
            )

            # Convert to NumPy array
            image = np.asarray(
                img,
                dtype=np.float32
            )

        return image

    except Exception as e:

        print(
            "Error loading image:",
            path
        )

        # Return correctly shaped fallback image
        return np.zeros(
            (224, 224, 3),
            dtype=np.float32
        )


# ============================================================
# TENSORFLOW IMAGE LOADER
# ============================================================

def load_image(path, label):

    image = tf.numpy_function(
        load_image_with_pil,
        [path],
        tf.float32
    )

    # IMPORTANT:
    # Force TensorFlow to know the exact shape
    image.set_shape(
        [224, 224, 3]
    )

    # MobileNetV2 preprocessing
    image = preprocess_input(
        image
    )

    return image, label


# ============================================================
# TRAIN DATASET
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

train_ds = train_ds.shuffle(
    buffer_size=min(
        500,
        len(train_paths)
    ),
    seed=SEED
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=1
)

train_ds = train_ds.batch(
    BATCH_SIZE
)

train_ds = train_ds.prefetch(1)


# ============================================================
# VALIDATION DATASET
# ============================================================

val_ds = tf.data.Dataset.from_tensor_slices(
    (
        val_paths,
        val_labels
    )
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=1
)

val_ds = val_ds.batch(
    BATCH_SIZE
)

val_ds = val_ds.prefetch(1)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.1
    ),

    layers.RandomZoom(
        0.1
    )
])


# ============================================================
# LOAD MOBILENETV2
# ============================================================

print()
print("=" * 60)
print("LOADING MOBILENETV2")
print("=" * 60)

base_model = MobileNetV2(

    input_shape=(
        224,
        224,
        3
    ),

    include_top=False,

    weights="imagenet"
)


# Freeze pretrained model
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        224,
        224,
        3
    )
)


# Data augmentation
x = data_augmentation(
    inputs
)


# MobileNetV2
x = base_model(
    x,
    training=False
)


# Convert feature maps to one vector
x = layers.GlobalAveragePooling2D()(x)


# Dropout
x = layers.Dropout(
    0.3
)(x)


# Dense layer
x = layers.Dense(
    64,
    activation="relu"
)(x)


# More dropout
x = layers.Dropout(
    0.2
)(x)


# Binary classification
# 0 = Fresh
# 1 = Spoiled

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


# Create model
model = models.Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 60)
print("MODEL")
print("=" * 60)

model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

print()

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 60)
print("SAVING MODEL")
print("=" * 60)

model.save(
    MODEL_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)

print()

print(
    "Model saved at:"
)

print(
    MODEL_PATH
)

print()

print("Class 0 = Fresh")
print("Class 1 = Spoiled")

print()
print("=" * 60)