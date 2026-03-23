import os
import requests

real_dir = r"c:\STUDY MATERIALS\MINI PROJECT\data\test\real"
url = "http://localhost:8000/api/predict"

total = 0
correct = 0
files = [f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:15]

for file in files:
    path = os.path.join(real_dir, file)
    with open(path, "rb") as f:
        try:
            res = requests.post(url, files={"file": (file, f, "image/jpeg")}).json()
            print(f"{file} -> {res['prediction']} (Conf: {res['confidence']:.3f})")
            if res['prediction'] == 'REAL':
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error on {file}: {e}")

if total > 0:
    print(f"Accuracy on {total} real images: {correct}/{total} ({(correct/total)*100:.1f}%)")
