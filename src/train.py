"""
Driver Drowsiness Detection - MobileNetV2 Transfer Learning
Two-phase training: Frozen base + Fine-tuning
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2

# ==================== CONFIGURATION ====================

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_PHASE1 = 20      # Frozen base training
EPOCHS_PHASE2 = 15      # Fine-tuning
SEED        = 42

DATASET_ROOT = "dataset"   # Change if needed
TRAIN_DIR    = os.path.join(DATASET_ROOT, "train")
VAL_DIR      = os.path.join(DATASET_ROOT, "val")

# If no separate val folder, use validation_split
USE_VALIDATION_SPLIT = False

CLASS_NAMES = ['Closed_Eyes', 'No_Yawn', 'Open_Eyes', 'Yawn']

SAVE_DIR          = "models"
BEST_MODEL_PATH   = os.path.join(SAVE_DIR, "best_model.keras")
FINAL_MODEL_PATH  = os.path.join(SAVE_DIR, "final_model.keras")

os.makedirs(SAVE_DIR, exist_ok=True)

# Set seeds
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"Classes: {CLASS_NAMES}\n")

# ==================== DATA GENERATORS ====================

train_datagen = ImageDataGenerator(
    rescale            = 1./255,
    rotation_range     = 10,
    width_shift_range  = 0.08,
    height_shift_range = 0.08,
    shear_range        = 0.08,
    zoom_range         = 0.10,
    horizontal_flip    = True,
    brightness_range   = [0.90, 1.10],
    fill_mode          = 'nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

if not USE_VALIDATION_SPLIT:
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size   = (IMG_SIZE, IMG_SIZE),
        batch_size    = BATCH_SIZE,
        class_mode    = 'categorical',
        classes       = CLASS_NAMES,
        shuffle       = True,
        seed          = SEED
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size   = (IMG_SIZE, IMG_SIZE),
        batch_size    = BATCH_SIZE,
        class_mode    = 'categorical',
        classes       = CLASS_NAMES,
        shuffle       = False,
        seed          = SEED
    )
else:
    full_datagen = ImageDataGenerator(
        rescale            = 1./255,
        rotation_range     = 10,
        width_shift_range  = 0.08,
        height_shift_range = 0.08,
        shear_range        = 0.08,
        zoom_range         = 0.10,
        horizontal_flip    = True,
        validation_split   = 0.20,
        fill_mode          = 'nearest'
    )

    train_gen = full_datagen.flow_from_directory(
        DATASET_ROOT,
        target_size   = (IMG_SIZE, IMG_SIZE),
        batch_size    = BATCH_SIZE,
        class_mode    = 'categorical',
        classes       = CLASS_NAMES,
        subset        = 'training',
        shuffle       = True,
        seed          = SEED
    )

    val_gen = full_datagen.flow_from_directory(
        DATASET_ROOT,
        target_size   = (IMG_SIZE, IMG_SIZE),
        batch_size    = BATCH_SIZE,
        class_mode    = 'categorical',
        classes       = CLASS_NAMES,
        subset        = 'validation',
        shuffle       = False,
        seed          = SEED
    )

print(f"Training samples : {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Class indices     : {train_gen.class_indices}\n")

# ==================== PHASE 1: FROZEN BASE ====================

print("=" * 60)
print("PHASE 1: Training with FROZEN MobileNetV2 base")
print("=" * 60)

# Load MobileNetV2 pre-trained on ImageNet
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze the base model
base_model.trainable = False

# Build model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(len(CLASS_NAMES), activation='softmax')
])

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()

callbacks_phase1 = [
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor      = 'val_accuracy',
        save_best_only = True,
        mode         = 'max',
        verbose      = 1
    ),
    EarlyStopping(
        monitor          = 'val_loss',
        patience         = 5,
        restore_best_weights = True,
        verbose          = 1
    ),
    ReduceLROnPlateau(
        monitor   = 'val_loss',
        factor    = 0.5,
        patience  = 3,
        min_lr    = 1e-6,
        verbose   = 1
    )
]

history_phase1 = model.fit(
    train_gen,
    epochs           = EPOCHS_PHASE1,
    validation_data  = val_gen,
    callbacks        = callbacks_phase1,
    verbose          = 1
)

print(f"\nPhase 1 complete. Best model saved to {BEST_MODEL_PATH}")

# ==================== PHASE 2: FINE-TUNING ====================

print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning UNFROZEN layers")
print("=" * 60)

# Unfreeze the base model for fine-tuning
base_model.trainable = True

# Fine-tune from this layer onwards (adjust as needed)
# Freeze all layers except the last 30
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Fine-tuning from layer {fine_tune_at} onwards")
print(f"Trainable variables: {len(model.trainable_variables)}")

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=0.00003),  # 10x lower
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

callbacks_phase2 = [
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor      = 'val_accuracy',
        save_best_only = True,
        mode         = 'max',
        verbose      = 1
    ),
    EarlyStopping(
        monitor          = 'val_loss',
        patience         = 8,
        restore_best_weights = True,
        verbose          = 1
    ),
    ReduceLROnPlateau(
        monitor   = 'val_loss',
        factor    = 0.5,
        patience  = 4,
        min_lr    = 1e-7,
        verbose   = 1
    )
]

history_phase2 = model.fit(
    train_gen,
    epochs           = EPOCHS_PHASE1 + EPOCHS_PHASE2,
    initial_epoch    = len(history_phase1.history['loss']),
    validation_data  = val_gen,
    callbacks        = callbacks_phase2,
    verbose          = 1
)

# Save final model
model.save(FINAL_MODEL_PATH)
print(f"\nFinal model saved to {FINAL_MODEL_PATH}")
print(f"Best model saved to {BEST_MODEL_PATH}")

# Final evaluation
val_loss, val_acc = model.evaluate(val_gen)
print(f"\nFinal Validation → Loss: {val_loss:.4f} - Accuracy: {val_acc*100:.2f}%")

# Save training history
import json
history_dict = {
    'phase1': {k: [float(v) for v in history_phase1.history[k]] for k in history_phase1.history.keys()},
    'phase2': {k: [float(v) for v in history_phase2.history[k]] for k in history_phase2.history.keys()}
}
with open('training_history.json', 'w') as f:
    json.dump(history_dict, f)
print("Training history saved to training_history.json")
