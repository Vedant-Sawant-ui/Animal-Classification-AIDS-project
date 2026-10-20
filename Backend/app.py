import json
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)

BACKEND_DIR = Path(__file__).resolve().parent

# Load the trained model
model = tf.keras.models.load_model(BACKEND_DIR / "Model" / "best_model.h5")

breed_model_path = BACKEND_DIR / "Model" / "pet_breed_model.keras"
breed_classes_path = BACKEND_DIR / "Model" / "pet_breed_classes.json"
breed_model = None
breed_classes = {"breeds": [], "animals": []}
if breed_model_path.is_file() and breed_classes_path.is_file():
    breed_model = tf.keras.models.load_model(breed_model_path)
    with breed_classes_path.open(encoding="utf-8") as class_file:
        breed_classes = json.load(class_file)
    if len(breed_classes["breeds"]) != breed_model.output_shape[-1]:
        raise ValueError("Breed class metadata does not match the breed model output")

# IMPORTANT: This list MUST match the order Keras used during training
# This is the same list we used in predict.py when it was working correctly
CLASS_NAMES = [
    "cat",
    "chimpanzee",
    "cow",
    "crow",
    "deer",
    "dog",
    "dolphin",
    "eagle",
    "goat",
    "horse",
    "leopard",
    "lion",
    "owl",
    "parrot",
    "pig",
    "shark",
    "sheep",
    "tiger",
    "whale",
    "wolf",
]


IMG_SIZE = (224, 224)

@app.route("/")
def home():
    return jsonify({"message": "Animal Classification API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image received"}), 400

    file = request.files["image"]

    # Load & preprocess image (SAME as training & predict.py)
    img = Image.open(file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0       # rescale to [0,1]
    img = np.expand_dims(img, axis=0) # shape (1, 224, 224, 3)

    preds = model.predict(img, verbose=0)[0]
    index = int(np.argmax(preds))
    confidence = float(preds[index]) * 100.0

    predicted_class = CLASS_NAMES[index]
    predicted_breed = None
    breed_confidence = None
    if breed_model is not None and predicted_class in ("cat", "dog"):
        breed_predictions = breed_model.predict(img * 2.0 - 1.0, verbose=0)[0]
        matching_indices = [
            class_index
            for class_index, animal in enumerate(breed_classes["animals"])
            if animal == predicted_class
        ]
        if matching_indices:
            matching_predictions = breed_predictions[matching_indices]
            best_match = int(np.argmax(matching_predictions))
            breed_index = matching_indices[best_match]
            species_total = float(np.sum(matching_predictions))
            predicted_breed = breed_classes["breeds"][breed_index]
            breed_confidence = float(matching_predictions[best_match] / species_total) * 100.0

    return jsonify({
        "animal": predicted_class,
        "confidence": round(confidence, 2),
        "breed": predicted_breed,
        "breed_confidence": round(breed_confidence, 2) if breed_confidence is not None else None,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
