import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# ==========================
# PATHS
# ==========================
base_data_dir = r"D:\AI PROJECTS\Animal_Classification\Data"

train_dir = os.path.join(base_data_dir, "Train")
val_dir   = os.path.join(base_data_dir, "Val")
test_dir  = os.path.join(base_data_dir, "Test")


# ==========================
# BASIC SETTINGS
# ==========================
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 20   # increase later if needed


# ==========================
# DATA LOADERS
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_gen.num_classes
print("Number of classes:", num_classes)
print("Class indices:", train_gen.class_indices)


# ==========================
# MODEL: MobileNetV2
# ==========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # freeze feature extractor

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================
# CALLBACKS
# ==========================
os.makedirs("Model/saved", exist_ok=True)

checkpoint_path = "Model/saved/best_model.h5"

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
]


# ==========================
# TRAIN MODEL
# ==========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ==========================
# PLOT ACCURACY & LOSS
# ==========================
plt.figure()
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("Accuracy")
plt.savefig("Model/saved/accuracy_plot.png")

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.savefig("Model/saved/loss_plot.png")


# ==========================
# EVALUATE ON TEST SET
# ==========================
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy = {test_acc:.4f}")


# ==========================
# SAVE FINAL MODEL
# ==========================
final_model_path = "Model/saved/animal_classifier_mobilenet.h5"
model.save(final_model_path)

print("\nModel saved successfully!")
print("Best model saved at:", checkpoint_path)
