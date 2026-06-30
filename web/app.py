import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow import keras

# ---------------------------------------
# Flask App
# ---------------------------------------

app = Flask(__name__)

# ---------------------------------------
# Paths
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------------------
# Load Model
# ---------------------------------------

model = keras.models.load_model("models/ai_detector.keras")

IMAGE_SIZE = (224, 224)

# ---------------------------------------
# Home Page
# ---------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------
# Display Uploaded Image
# ---------------------------------------

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ---------------------------------------
# Prediction
# ---------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "No file selected."

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    file.save(filepath)

    image = keras.utils.load_img(
        filepath,
        target_size=IMAGE_SIZE
    )

    image_array = keras.utils.img_to_array(image)

    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    confidence = float(prediction[0][0])

    if confidence >= 0.5:
        result = "REAL IMAGE"
        confidence = confidence * 100
    else:
        result = "AI GENERATED"
        confidence = (1 - confidence) * 100

    return render_template(
        "index.html",
        prediction=result,
        confidence=f"{confidence:.2f}",
        image=file.filename
    )


# ---------------------------------------
# Run
# ---------------------------------------

if __name__ == "__main__":
    app.run(debug=True)