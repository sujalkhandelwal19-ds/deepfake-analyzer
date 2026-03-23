import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, Xception
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

class DeepfakeDetector:
    
    def __init__(self, train_dir, test_dir, img_size=(224, 224), batch_size=32):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.results = {}
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f'results_{self.timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)
        
        print("="*70)
        print("DEEPFAKE DETECTION SYSTEM - PROFESSIONAL EDITION")
        print("="*70)
        
    def prepare_data(self):
        print("\nStep 1: Preparing Dataset...")
        print("-" * 70)
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.2,
            shear_range=0.15,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=True,
            seed=42
        )
        
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Classes detected: {self.train_generator.class_indices}")
        
        total = self.train_generator.samples
        fake_count = len(os.listdir(os.path.join(self.train_dir, 'fake')))
        real_count = len(os.listdir(os.path.join(self.train_dir, 'real')))
        
        self.class_weights = {
            0: total / (2.0 * fake_count),
            1: total / (2.0 * real_count)
        }
        print(f"Class weights: {self.class_weights}")
        
    def build_advanced_model(self):
        print("\nStep 2: Building Advanced Architecture...")
        print("-" * 70)
        
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        base_efficient = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        base_efficient.trainable = False
        x1 = layers.GlobalAveragePooling2D()(base_efficient.output)
        
        base_xception = Xception(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        base_xception.trainable = False
        x2 = layers.GlobalAveragePooling2D()(base_xception.output)
        
        combined = layers.Concatenate()([x1, x2])
        
        attention = layers.Dense(512, activation='relu')(combined)
        attention = layers.Dense(combined.shape[-1], activation='sigmoid')(attention)
        attended = layers.Multiply()([combined, attention])
        
        x = layers.BatchNormalization()(attended)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='Deepfake_Detector_Ensemble')
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print("Model architecture built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        with open(f'{self.results_dir}/model_summary.txt', 'w', encoding='utf-8') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    def train(self, epochs=30, fine_tune=True):
        print(f"\nStep 3: Training Model...")
        print("-" * 70)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'{self.results_dir}/best_model.h5',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=f'{self.results_dir}/logs',
                histogram_freq=1
            )
        ]
        
        print("\nPhase 1: Training with frozen base models...")
        self.history = self.model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=epochs // 2,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        if fine_tune:
            print("\nPhase 2: Fine-tuning last layers...")
            
            for layer in self.model.layers:
                if 'efficient' in layer.name or 'xception' in layer.name:
                    layer.trainable = True
                    if isinstance(layer, keras.layers.BatchNormalization):
                        layer.trainable = False
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')
                ]
            )
            
            history_fine = self.model.fit(
                self.train_generator,
                validation_data=self.val_generator,
                epochs=epochs // 2,
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            for key in self.history.history.keys():
                self.history.history[key].extend(history_fine.history[key])
        
        print("\nTraining completed successfully!")
        
    def plot_comprehensive_results(self):
        print("\nStep 4: Generating Visualizations...")
        print("-" * 70)
        
        fig = plt.figure(figsize=(20, 12))
        
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(self.history.history['accuracy'], label='Train', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(self.history.history['loss'], label='Train', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(self.history.history['precision'], label='Train', linewidth=2)
        ax3.plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        ax3.set_title('Precision', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(self.history.history['recall'], label='Train', linewidth=2)
        ax4.plot(self.history.history['val_recall'], label='Validation', linewidth=2)
        ax4.set_title('Recall', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend(loc='lower right')
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(self.history.history['auc'], label='Train', linewidth=2)
        ax5.plot(self.history.history['val_auc'], label='Validation', linewidth=2)
        ax5.set_title('AUC Score', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('AUC')
        ax5.legend(loc='lower right')
        ax5.grid(True, alpha=0.3)
        
        ax6 = plt.subplot(2, 3, 6)
        if 'lr' in self.history.history:
            ax6.plot(self.history.history['lr'], linewidth=2, color='orange')
            ax6.set_title('Learning Rate', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Learning Rate')
            ax6.set_yscale('log')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Learning Rate\nNot Logged', 
                    ha='center', va='center', fontsize=12)
            ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Saved: training_metrics.png")
        plt.close()
        
    def evaluate_model(self):
        print("\nStep 5: Evaluating Model Performance...")
        print("-" * 70)
        
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred_proba = predictions.flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        y_true = self.test_generator.classes
        
        print("\n" + "="*70)
        print("CLASSIFICATION REPORT")
        print("="*70)
        report = classification_report(y_true, y_pred, 
                                       target_names=['Fake', 'Real'],
                                       digits=4)
        print(report)
        
        with open(f'{self.results_dir}/classification_report.txt', 'w') as f:
            f.write(report)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                    xticklabels=['Fake', 'Real'],
                    yticklabels=['Fake', 'Real'],
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 16})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"Saved: confusion_matrix.png")
        plt.close()
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        print(f"Saved: roc_curve.png")
        plt.close()
        
        accuracy = np.mean(y_pred == y_true)
        precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
        recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_score': float(roc_auc),
            'confusion_matrix': cm.tolist()
        }
        
        with open(f'{self.results_dir}/metrics.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print("\n" + "="*70)
        print("FINAL METRICS")
        print("="*70)
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1-Score:  {f1*100:.2f}%")
        print(f"AUC Score: {roc_auc*100:.2f}%")
        print("="*70)
        
    def visualize_predictions(self, num_samples=20):
        print(f"\nStep 6: Visualizing Sample Predictions...")
        print("-" * 70)
        
        self.test_generator.reset()
        x_batch, y_batch = next(self.test_generator)
        predictions = self.model.predict(x_batch[:num_samples], verbose=0)
        
        rows = 4
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(x_batch))):
            img = x_batch[i]
            true_label = 'Real' if y_batch[i] == 1 else 'Fake'
            pred_prob = predictions[i][0]
            pred_label = 'Real' if pred_prob > 0.5 else 'Fake'
            confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob
            
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(
                f'True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)',
                fontsize=10, fontweight='bold', color=color
            )
        
        plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/sample_predictions.png', dpi=300, bbox_inches='tight')
        print(f"Saved: sample_predictions.png")
        plt.close()
        
    def predict_single_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, self.img_size)
        img_array = np.expand_dims(img_resized / 255.0, axis=0)
        
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        label = "REAL" if prediction > 0.5 else "FAKE"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        title_color = 'green' if label == 'REAL' else 'red'
        plt.title(f'Prediction: {label}\nConfidence: {confidence*100:.2f}%', 
                 fontsize=18, fontweight='bold', color=title_color, pad=20)
        plt.tight_layout()
        plt.show()
        
        return label, confidence
    
    def save_final_model(self):
        print("\nStep 7: Saving Model and Documentation...")
        print("-" * 70)
        
        model_path = f'{self.results_dir}/deepfake_detector_final.h5'
        self.model.save(model_path)
        print(f"Model saved: {model_path}")
        
        readme_content = f"""# Deepfake Detection System
## Project Overview
Advanced deepfake detection using ensemble deep learning with attention mechanism.

## Model Architecture
- Base Models: EfficientNetB3 + Xception (Ensemble)
- Additional Layers: Attention mechanism, BatchNormalization, Dropout
- Total Parameters: {self.model.count_params():,}

## Performance Metrics
- Accuracy: {self.results['accuracy']*100:.2f}%
- Precision: {self.results['precision']*100:.2f}%
- Recall: {self.results['recall']*100:.2f}%
- F1-Score: {self.results['f1_score']*100:.2f}%
- AUC Score: {self.results['auc_score']*100:.2f}%

## Dataset Information
- Training Samples: {self.train_generator.samples}
- Test Samples: {self.test_generator.samples}
- Image Size: {self.img_size}
- Batch Size: {self.batch_size}

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

print(f"Prediction: {{label}} (Confidence: {{confidence*100:.2f}}%)")
```

## Technologies Used
- TensorFlow/Keras
- EfficientNet & Xception (Transfer Learning)
- Python, NumPy, OpenCV
- Matplotlib, Seaborn

## Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        with open(f'{self.results_dir}/README.md', 'w') as f:
            f.write(readme_content)
        
        print(f"Documentation saved: README.md")
        print(f"\nAll results saved in: {self.results_dir}/")


def main():
    
    TRAIN_DIR = 'data/train'
    TEST_DIR = 'data/test'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    
    detector = DeepfakeDetector(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    detector.prepare_data()
    detector.build_advanced_model()
    detector.train(epochs=EPOCHS, fine_tune=True)
    detector.plot_comprehensive_results()
    detector.evaluate_model()
    detector.visualize_predictions(num_samples=20)
    detector.save_final_model()
    
    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print(f"Check results in: {detector.results_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()