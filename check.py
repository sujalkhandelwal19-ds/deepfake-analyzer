import asyncio
from fastapi import UploadFile
import io
import os
import sys

sys.path.append(r"c:\STUDY MATERIALS\MINI PROJECT")
from backend.main import load_model, predict

async def test():
    load_model()
    with open(r"c:\STUDY MATERIALS\MINI PROJECT\data\test\fake\fake_041594.jpg", "rb") as f:
        content = f.read()
    
    file = UploadFile(filename="fake.jpg", file=io.BytesIO(content))
    async def mock_read():
        return content
    file.read = mock_read
    file.content_type = "image/jpeg"
    
    try:
        res = await predict(file)
        print(res.body)
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
