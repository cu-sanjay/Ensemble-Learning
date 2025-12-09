import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define constants
MODEL_PATH = "models/lung_disease_detection_model.h5"
IMAGE_SIZE = 150
CLASS_NAMES = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")

model = load_model(MODEL_PATH)

def predict_image(image_path):
    """Predicts the lung disease category for a given image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Load and preprocess the image
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Print and return result
    result = CLASS_NAMES[predicted_class]
    print(f"✅ Predicted Class: {result} (Confidence: {np.max(predictions) * 100:.2f}%)")
    return result

# Example usage
if __name__ == "__main__":
    IMAGE_PATH = "data/Lung Disease Dataset/test/Corona Virus Disease/00030375_007.png"
    predict_image(IMAGE_PATH)