from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import os
import sqlite3
import pandas as pd
from data_preparer import MODEL_CONFIG, VEHICLE_CLASSES, DATABASE_CONFIG, LOGGING_CONFIG  # Sửa import
import logging

# Cấu hình logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI app
app = FastAPI(title="Traffic Detection API", description="API for vehicle detection and classification")

# Tải mô hình YOLO
try:
    model = YOLO(MODEL_CONFIG['model_name'])
    logger.info(f"Đã tải mô hình YOLO: {MODEL_CONFIG['model_name']}")
except Exception as e:
    logger.error(f"Không thể tải mô hình YOLO: {e}")
    raise Exception(f"Không thể tải mô hình YOLO: {e}")

def process_frame(frame):
    """Xử lý một frame với YOLO và trả về frame đã được chú thích."""
    results = model(frame, conf=MODEL_CONFIG['confidence_threshold'], iou=MODEL_CONFIG['iou_threshold'])
    annotated_frame = results[0].plot()  # Vẽ bounding box và nhãn
    return annotated_frame

def save_detection(class_name, confidence, bbox):
    """Lưu thông tin phát hiện vào cơ sở dữ liệu."""
    conn = sqlite3.connect(DATABASE_CONFIG['path'])
    c = conn.cursor()
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

        if file.content_type.startswith('image/'):
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.error("Không thể giải mã ảnh")
                raise HTTPException(status_code=400, detail="File ảnh không hợp lệ")
            results = model(frame, conf=MODEL_CONFIG['confidence_threshold'], iou=MODEL_CONFIG['iou_threshold'])
            annotated_frame = results[0].plot()
            # Lưu các phát hiện vào cơ sở dữ liệu
            for det in results[0].boxes:
                if det.conf > MODEL_CONFIG['confidence_threshold']:
                    cls = int(det.cls)
                    label = model.names[cls]
                    if label in VEHICLE_CLASSES.values():
                        bbox = det.xywh[0].tolist()
                        save_detection(label, det.conf.item(), bbox)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            return StreamingResponse(BytesIO(buffer), media_type="image/jpeg")

        elif file.content_type.startswith('video/'):
            temp_file = "temp_video.mp4"
            with open(temp_file, "wb") as f:
                f.write(contents)
            cap = cv2.VideoCapture(temp_file)
            if not cap.isOpened():
                logger.error("Không thể mở file video")
                raise HTTPException(status_code=400, detail="File video không hợp lệ")
            ret, frame = cap.read()
            if not ret:
                logger.error("Không thể đọc frame video")
                cap.release()
                os.remove(temp_file)
                raise HTTPException(status_code=400, detail="Video rỗng hoặc hỏng")
            results = model(frame, conf=MODEL_CONFIG['confidence_threshold'], iou=MODEL_CONFIG['iou_threshold'])
            annotated_frame = results[0].plot()
            # Lưu các phát hiện vào cơ sở dữ liệu
            for det in results[0].boxes:
                if det.conf > MODEL_CONFIG['confidence_threshold']:
                    cls = int(det.cls)
                    label = model.names[cls]
                    if label in VEHICLE_CLASSES.values():
                        bbox = det.xywh[0].tolist()
                        save_detection(label, det.conf.item(), bbox)
            cap.release()
            os.remove(temp_file)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            return StreamingResponse(BytesIO(buffer), media_type="image/jpeg")

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
        if df.empty:
            logger.warning("Không có dữ liệu phát hiện để tạo báo cáo")
            return JSONResponse(content={"message": "Không có dữ liệu phát hiện"}, status_code=200)
        return JSONResponse(content=df.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Lỗi khi tạo báo cáo: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo báo cáo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)