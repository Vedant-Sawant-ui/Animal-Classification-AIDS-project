import tensorflow as tf
import numpy as np
import argparse
import os
from tensorflow.keras.preprocessing import image

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = r"Model/saved/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# IMPORTANT: class names must match train_gen.class_indices order.
# Keras sorts folders alphabetically, so for your 20 classes:
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

# ===============================
# PREDICT FUNCTION
# ===============================
def predict_image(img_path):
    # Load & resize
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)

    # Same preprocessing as training: rescale to [0,1]
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][class_index]) * 100.0

    result = CLASS_NAMES[class_index]
    return result, confidence


# ===============================
# MAIN SCRIPT
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the image file")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("Error: Image file not found!")
        raise SystemExit

    pred_class, conf = predict_image(args.image)

    print(f"\nPredicted Animal: {pred_class}")
    print(f"Confidence: {conf:.2f}%")
