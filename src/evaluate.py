import tensorflow as tf
from tensorflow import keras

# -----------------------------
# Load the trained model
# -----------------------------
model = keras.models.load_model("models/ai_detector.keras")

# -----------------------------
# Settings
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# -----------------------------
# Load test dataset
# -----------------------------
test_dataset = keras.utils.image_dataset_from_directory(
    "processed_data/test",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# Evaluate the model
# -----------------------------
loss, accuracy = model.evaluate(test_dataset)

print("\n==========================")
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")
print("==========================")