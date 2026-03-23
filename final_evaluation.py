import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = 'results_20260212_224132/best_model.h5'
TEST_DIR = 'data/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

print("Preparing test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"Test samples: {test_generator.samples}")

print("\nEvaluating model...")
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
y_pred_proba = predictions.flatten()
y_pred = (y_pred_proba > 0.5).astype(int)
y_true = test_generator.classes

print("\n" + "="*70)
print("CLASSIFICATION REPORT - FULLY TRAINED MODEL")
print("="*70)
report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'], digits=4)
print(report)

with open('results_20260212_224132/classification_report.txt', 'w') as f:
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'], cbar_kws={'label': 'Count'}, annot_kws={'size': 16})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('results_20260212_224132/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix.png")

fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results_20260212_224132/roc_curve.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curve.png")

accuracy = np.mean(y_pred == y_true)
precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

results = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auc_score': float(roc_auc)
}

with open('results_20260212_224132/metrics.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\n" + "="*70)
print("FINAL METRICS - COMPLETE TRAINING")
print("="*70)
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall:    {recall*100:.2f}%")
print(f"F1-Score:  {f1*100:.2f}%")
print(f"AUC Score: {roc_auc*100:.2f}%")
print("="*70)