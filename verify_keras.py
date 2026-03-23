import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

REAL_DIR = r"c:\STUDY MATERIALS\MINI PROJECT\data\test\real"

class CustomDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

print("Loading model...")
model = keras.models.load_model(r"c:\STUDY MATERIALS\MINI PROJECT\results_20260311_210339\deepfake_detector_final.h5", custom_objects={'Dense': CustomDense})
print("Model loaded.")

files = [f for f in os.listdir(REAL_DIR) if f.endswith('.jpg')][:50]
correct = 0

for file in files:
    img = cv2.imread(os.path.join(REAL_DIR, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    
    pred = model.predict(img_array, verbose=0)[0][0]
    is_real = pred > 0.5
    if is_real:
        correct += 1

print(f"\nFinal Accuracy on {len(files)} real images: {correct}/{len(files)} ({(correct/len(files))*100:.1f}%)")

try:
    with open(r"c:\STUDY MATERIALS\MINI PROJECT\results_20260311_210339\metrics.json", "r") as f:
        metrics = json.load(f)
        print("Model training metrics:", metrics)
except:
    pass
