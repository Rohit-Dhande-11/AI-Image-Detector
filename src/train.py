import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ==========================================
# Settings
# ==========================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

train_dataset = keras.utils.image_dataset_from_directory(
    "processed_data/train",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = keras.utils.image_dataset_from_directory(
    "processed_data/validation",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClasses:", train_dataset.class_names)

# ==========================================
# Improve Performance
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# ==========================================
# Data Augmentation
# ==========================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ==========================================
# Load EfficientNetB0
# ==========================================

base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# ==========================================
# Build Model
# ==========================================

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# Show Model Summary
# ==========================================

model.summary()

# ==========================================
# Callbacks
# ==========================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="models/ai_detector.keras",
    save_best_only=True,
    monitor="val_accuracy"
)

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# ==========================================
# Plot Accuracy
# ==========================================

plt.figure(figsize=(8,5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# Plot Loss
# ==========================================

plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

print("\n===================================")
print("Training Completed Successfully!")
print("Model saved in models/ai_detector.keras")
print("===================================")