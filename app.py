import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Thêm CORS
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import sqlite3
import pandas as pd
from datetime import datetime
from data_preparer import MODEL_CONFIG, DATABASE_CONFIG, LOGGING_CONFIG

# Tạo thư mục temp và logs nếu chưa tồn tại
os.makedirs("temp", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Cấu hình logging (sử dụng cấu hình từ data_preparer)
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI app
app = FastAPI(title="Traffic Detection API", description="API for vehicle detection and classification")

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000"],  # Cho phép frontend và Node.js server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tải mô hình YOLO
try:
    model_path = os.path.join(os.path.dirname(__file__), "models", MODEL_CONFIG['model_name'])
    if not os.path.exists(model_path):
        logger.error(f"File mô hình không tồn tại: {model_path}")
        raise Exception(f"File mô hình không tồn tại: {model_path}")
    model = YOLO(model_path)
    logger.info(f"Đã tải mô hình YOLO: {MODEL_CONFIG['model_name']}")
except Exception as e:
    logger.error(f"Không thể tải mô hình YOLO: {e}")
    raise Exception(f"Không thể tải mô hình YOLO: {e}")

def process_frame(frame):
    """Xử lý một frame với YOLO và trả về frame đã được chú thích."""
    results = model(frame, conf=MODEL_CONFIG['confidence_threshold'])
    annotated_frame = results[0].plot()  # Vẽ bounding box và nhãn
    return annotated_frame

def save_detection(class_name, confidence, bbox):
    """Lưu thông tin phát hiện vào cơ sở dữ liệu."""
    conn = sqlite3.connect(DATABASE_CONFIG['path'])
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            class_name TEXT,
            confidence REAL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL
        )
    ''')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''
        INSERT INTO detections (timestamp, class_name, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (timestamp, class_name, confidence, *bbox))
    conn.commit()
    conn.close()
    logger.info(f"Đã lưu phát hiện: {class_name}, confidence: {confidence}")

@app.post("/detect")
async def detect_vehicles(file: UploadFile = File(...)):
    """Phát hiện phương tiện trong ảnh hoặc video được tải lên."""
    if not file.content_type.startswith(('image/', 'video/')):
        logger.error(f"Loại file không hợp lệ: {file.content_type}")
        raise HTTPException(status_code=400, detail="File phải là ảnh hoặc video")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        detections = []

        if file.content_type.startswith('image/'):
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.error("Không thể giải mã ảnh")
                raise HTTPException(status_code=400, detail="File ảnh không hợp lệ")
            results = model(frame, conf=MODEL_CONFIG['confidence_threshold'])
            for det in results[0].boxes:
                if det.conf > MODEL_CONFIG['confidence_threshold']:
                    cls = int(det.cls)
                    label = model.names[cls]
                    if label in MODEL_CONFIG['vehicle_classes'].values():
                        bbox = det.xywh[0].tolist()
                        save_detection(label, det.conf.item(), bbox)
                        detections.append({
                            "class_name": label,
                            "confidence": det.conf.item(),
                            "bbox": bbox
                        })
            return JSONResponse(content={"detections": detections})

        elif file.content_type.startswith('video/'):
            temp_file = os.path.join("temp", "temp_video.mp4")  # Lưu vào thư mục temp
            with open(temp_file, "wb") as f:
                f.write(contents)
            cap = cv2.VideoCapture(temp_file)
            if not cap.isOpened():
                logger.error(f"Không thể mở file video: {temp_file}")
                os.remove(temp_file)
                raise HTTPException(status_code=400, detail="File video không hợp lệ")
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Không thể đọc frame video: {temp_file}")
                cap.release()
                os.remove(temp_file)
                raise HTTPException(status_code=400, detail="Video rỗng hoặc hỏng")
            results = model(frame, conf=MODEL_CONFIG['confidence_threshold'])
            for det in results[0].boxes:
                if det.conf > MODEL_CONFIG['confidence_threshold']:
                    cls = int(det.cls)
                    label = model.names[cls]
                    if label in MODEL_CONFIG['vehicle_classes'].values():
                        bbox = det.xywh[0].tolist()
                        save_detection(label, det.conf.item(), bbox)
                        detections.append({
                            "class_name": label,
                            "confidence": det.conf.item(),
                            "bbox": bbox
                        })
            cap.release()
            os.remove(temp_file)
            return JSONResponse(content={"detections": detections})

    except Exception as e:
        logger.error(f"Lỗi khi xử lý file: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý file: {str(e)}")

@app.get("/report")
async def get_vehicle_report():
    """Trả về báo cáo thống kê số lượng phương tiện theo loại."""
    try:
        conn = sqlite3.connect(DATABASE_CONFIG['path'])
        query = "SELECT class_name, COUNT(*) as count FROM detections GROUP BY class_name"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty or df['class_name'].isnull().all():
            logger.warning("Không có dữ liệu phát hiện hoặc dữ liệu không hợp lệ để tạo báo cáo")
            return JSONResponse(content={"message": "Không có dữ liệu phát hiện"}, status_code=200)
        return JSONResponse(content=df.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Lỗi khi tạo báo cáo: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo báo cáo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)