import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import os
from data_loader import load_dataset

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load dataset
X_train, Y_train, X_test, Y_test, X_val, Y_val = load_dataset()

# Define model architecture using transfer learning
def build_model(input_shape=(150, 150, 3), num_classes=5):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # Freeze all layers except last few for fine-tuning
    for layer in base_model.layers[:-10]:  
        layer.trainable = False

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# Build and compile the model
model = build_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)

# Save the trained model
model_save_path = "models/lung_disease_detection_model.h5"
os.makedirs("models", exist_ok=True)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")