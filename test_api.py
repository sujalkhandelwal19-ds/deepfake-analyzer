import requests
import json
import time

url = "http://localhost:8000/api/predict"
image_path = r"data\test\fake\fake_041594.jpg"

print("Waiting for server to be ready...")
for _ in range(5):
    try:
        requests.get("http://localhost:8000/")
        break
    except:
        time.sleep(1)

print(f"Testing API with {image_path}...")
try:
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    print(f"Response Body: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Test failed: {e}")
