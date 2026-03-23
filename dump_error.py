import traceback
from tensorflow import keras

try:
    keras.models.load_model(r'c:\STUDY MATERIALS\MINI PROJECT\results_20260311_210339\deepfake_detector_final.h5')
except Exception as e:
    with open('err.txt', 'w') as f:
        traceback.print_exc(file=f)
