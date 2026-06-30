import tensorflow as tf
from tensorflow import keras
import numpy as np

# -----------------------------
# Settings
# -----------------------------
IMAGE_SIZE = (224, 224)

# -----------------------------
# Load Model
# -----------------------------
model = keras.models.load_model("models/ai_detector.keras")

# -----------------------------
# Image Path
# -----------------------------
IMAGE_PATH = "test_images/test.png"

# -----------------------------
# Load Image
# -----------------------------
image = keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMAGE_SIZE
)

image_array = keras.utils.img_to_array(image)

image_array = np.expand_dims(image_array, axis=0)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(image_array)

confidence = prediction[0][0]

if confidence >= 0.5:
    print("\nPrediction : REAL IMAGE")
    print(f"Confidence : {confidence*100:.2f}%")

else:
    print("\nPrediction : AI GENERATED")
    print(f"Confidence : {(1-confidence)*100:.2f}%")