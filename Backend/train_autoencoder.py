import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "dataset")

IMG_SIZE = 128
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='input',   # 🔥 IMPORTANT FIX
    shuffle=True
)

# ---------------- AUTOENCODER ----------------

input_img = layers.Input(shape=(128,128,3))

# Encoder
x = layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D(2, padding='same')(x)

# Decoder
x = layers.Conv2D(64, 3, activation='relu', padding='same')(encoded)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(2)(x)
decoded = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)

autoencoder.compile(
    optimizer='adam',
    loss='mse'
)

autoencoder.fit(
    generator,
    epochs=20
)

# Save model
model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

autoencoder.save(os.path.join(model_dir, "autoencoder.h5"))

print("✅ Autoencoder trained and saved successfully")
