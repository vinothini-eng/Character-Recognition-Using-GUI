import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_PATH = "final_dataset"

IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 10

# -----------------------------
# Load Dataset
# -----------------------------

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    subset="training",
    shuffle=True
)

validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    subset="validation",
    shuffle=False
)

print("Class mapping:")
print(train_data.class_indices)

# -----------------------------
# CNN Model
# -----------------------------

model = Sequential([
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 1)
    ),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train
# -----------------------------

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# -----------------------------
# Save Model
# -----------------------------

model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.weights.h5")

print()
print("==============================")
print("MODEL TRAINING COMPLETED!")
print("model.json created")
print("model.weights.h5 created")
print("==============================")
