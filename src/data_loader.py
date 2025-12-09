import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Define dataset paths
DATASET_PATH = "data/Lung Disease Dataset"
IMAGE_SIZE = 150
LABELS = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

def load_images_from_folder(folder_path, label):
    """Load images from a folder, resize them, and assign labels."""
    images = []
    labels = []
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(img)
            labels.append(label)

    return images, labels

def load_dataset():
    """Loads training, validation, and testing datasets."""
    X_train, Y_train, X_test, Y_test, X_val, Y_val = [], [], [], [], [], []
    
    for label in LABELS:
        # Load Training Data
        train_path = os.path.join(DATASET_PATH, "train", label)
        images, labels = load_images_from_folder(train_path, LABELS.index(label))
        X_train.extend(images)
        Y_train.extend(labels)

        # Load Testing Data
        test_path = os.path.join(DATASET_PATH, "test", label)
        images, labels = load_images_from_folder(test_path, LABELS.index(label))
        X_test.extend(images)
        Y_test.extend(labels)

        # Load Validation Data
        val_path = os.path.join(DATASET_PATH, "val", label)
        images, labels = load_images_from_folder(val_path, LABELS.index(label))
        X_val.extend(images)
        Y_val.extend(labels)

    # Convert to numpy arrays
    X_train, Y_train = np.array(X_train), np.array(Y_train)
    X_test, Y_test = np.array(X_test), np.array(Y_test)
    X_val, Y_val = np.array(X_val), np.array(Y_val)

    # Shuffle training data
    X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

    # Normalize images (scale pixel values between 0 and 1)
    X_train, X_test, X_val = X_train / 255.0, X_test / 255.0, X_val / 255.0

    # Convert labels to categorical format
    Y_train = to_categorical(Y_train, num_classes=len(LABELS))
    Y_test = to_categorical(Y_test, num_classes=len(LABELS))
    Y_val = to_categorical(Y_val, num_classes=len(LABELS))

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test, X_val, Y_val = load_dataset()
    print(f"Training data: {X_train.shape}, {Y_train.shape}")
    print(f"Testing data: {X_test.shape}, {Y_test.shape}")
    print(f"Validation data: {X_val.shape}, {Y_val.shape}")