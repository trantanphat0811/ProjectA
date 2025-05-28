cat <<EOL > app.py
# app.py - FastAPI web interface for traffic detection visualization
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/detect")
async def detect_vehicles(file: UploadFile = File(...)):
    """Detect vehicles in an uploaded video."""
    model = YOLO('yolov8n.pt')
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED))
    output_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        output_frames.append(results[0].plot())

    cap.release()
    _, buffer = cv2.imencode('.jpg', output_frames[-1])
    return StreamingResponse(BytesIO(buffer), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOL