# Deepfake Detection System
## Project Overview
Advanced deepfake detection using ensemble deep learning with attention mechanism.

## Model Architecture
- Base Models: EfficientNetB3 + Xception (Ensemble)
- Additional Layers: Attention mechanism, BatchNormalization, Dropout
- Total Parameters: 37,336,408

## Performance Metrics
- Accuracy: 73.33%
- Precision: 76.81%
- Recall: 70.12%
- F1-Score: 73.32%
- AUC Score: 80.31%

## Dataset Information
- Training Samples: 72328
- Test Samples: 21776
- Image Size: (224, 224)
- Batch Size: 32

## Training Details
- Optimizer: Adam (lr=0.001, fine-tune lr=0.0001)
- Loss Function: Binary Crossentropy
- Data Augmentation: Rotation, Shift, Flip, Zoom, Brightness
- Class Weighting: Applied for balanced learning
- Early Stopping: Patience=7
- Learning Rate Reduction: Factor=0.3, Patience=4

## Files Generated
1. best_model.h5 - Best model checkpoint
2. deepfake_detector_final.h5 - Final trained model
3. training_metrics.png - Training history plots
4. confusion_matrix.png - Confusion matrix visualization
5. roc_curve.png - ROC curve
6. sample_predictions.png - Sample prediction visualizations
7. classification_report.txt - Detailed classification report
8. metrics.json - Performance metrics in JSON format
9. model_summary.txt - Model architecture summary

## Usage
```python
from tensorflow import keras

model = keras.models.load_model('deepfake_detector_final.h5')

import cv2
import numpy as np

img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img / 255.0, axis=0)

prediction = model.predict(img)[0][0]
label = "REAL" if prediction > 0.5 else "FAKE"
confidence = prediction if prediction > 0.5 else 1 - prediction

print(f"Prediction: {label} (Confidence: {confidence*100:.2f}%)")
```

## Technologies Used
- TensorFlow/Keras
- EfficientNet & Xception (Transfer Learning)
- Python, NumPy, OpenCV
- Matplotlib, Seaborn

## Generated: 2026-03-12 21:31:31
