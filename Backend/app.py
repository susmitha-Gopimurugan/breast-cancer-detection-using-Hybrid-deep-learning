import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
autoencoder_path = os.path.join(BASE_DIR, "model", "autoencoder.h5")
classifier_path = os.path.join(BASE_DIR, "model", "classifier.h5")

autoencoder = tf.keras.models.load_model(autoencoder_path, compile=False)
classifier = tf.keras.models.load_model(classifier_path, compile=False)

IMG_SIZE = 128

# Thresholds
RECON_THRESHOLD = 0.01
VARIANCE_MIN = 0.005
VARIANCE_MAX = 0.08

def preprocess(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file).convert("RGB")
    img = preprocess(image)

    # --------------------------
    # STEP 1: Statistical Check
    # --------------------------
    variance = float(np.var(img))  # Convert float32 -> float
    print("Image Variance:", variance)

    if variance < VARIANCE_MIN or variance > VARIANCE_MAX:
        return jsonify({
            "prediction": "No Cancer Detected",
            "confidence": 0.0
        })

    # --------------------------
    # STEP 2: Autoencoder Check
    # --------------------------
    reconstructed = autoencoder.predict(img)
    reconstruction_error = float(np.mean((img - reconstructed) ** 2))  # Convert float32 -> float
    print("Reconstruction Error:", reconstruction_error)

    if reconstruction_error > RECON_THRESHOLD:
        return jsonify({
            "prediction": "No Cancer Detected",
            "confidence": 0.0
        })

    # --------------------------
    # STEP 3: CNN Classification
    # --------------------------
    prediction = classifier.predict(img)[0]

    benign_prob = float(prediction[0])
    malignant_prob = float(prediction[1])

    if benign_prob > malignant_prob:
        label = "Benign"
        confidence = benign_prob * 100
    else:
        label = "Malignant"
        confidence = malignant_prob * 100

    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
