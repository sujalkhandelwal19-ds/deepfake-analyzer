import os
import io
import numpy as np
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = FastAPI(title="Deepfake Detection API", version="1.0")

# Enable CORS for local testing if frontend is served separately
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration — use paths relative to this file so it works on any OS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "results_20260311_210339", "deepfake_detector_final.h5")
)
IMG_SIZE = (224, 224)

# Global model variable
model = None

class CustomDense(keras.layers.Dense):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

@app.on_event("startup")
def load_model():
    """Load the deepfake detection model on application startup."""
    global model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = keras.models.load_model(MODEL_PATH, custom_objects={'Dense': CustomDense})
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Not raising an exception to still allow the app to start and show a friendly error later.

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess the image bytes to the format expected by the model."""
    try:
        # Load image with PIL
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (in case of RGBA/Grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert to numpy array
        img_array = np.array(img)
        
        # Resize using cv2 to match training format
        # If cv2 causes issues we can use PIL resize, but keeping it standard to training code
        img_resized = cv2.resize(img_array, IMG_SIZE)
        
        # Scale and add batch dimension
        img_scaled = np.expand_dims(img_resized / 255.0, axis=0)
        return img_scaled
    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """API endpoint to predict if an uploaded image is a deepfake."""
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="The demonstration model is currently not loaded. Please ensure the h5 model exists and restart the backend.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are supported.")
        
    try:
        # Read the file bytes
        contents = await file.read()
        
        # Preprocess
        processed_image = preprocess_image(contents)
        
        # Perform inference
        prediction_val = float(model.predict(processed_image, verbose=0)[0][0])
        
        # Post-process
        label = "REAL" if prediction_val > 0.5 else "FAKE"
        confidence = prediction_val if prediction_val > 0.5 else 1.0 - prediction_val
        
        return JSONResponse({
            "success": True,
            "prediction": label,
            "confidence": confidence,
            "raw_score": prediction_val
        })
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

# Mount frontend static directory if it exists
frontend_path = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Warning: Frontend directory not found at {frontend_path}")
