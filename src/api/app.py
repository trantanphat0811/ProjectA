import sys
import os
import logging
import cv2
import sqlite3
import numpy as np
from datetime import datetime
import pandas as pd
from io import BytesIO
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from src.core.speed_ditector import SpeedDetector
from src.utils.visualization import draw_results
from src.utils.video_processor import process_video
from src.core.config import (
    STORAGE_CONFIG, CAMERA_CONFIG, TRACKING_CONFIG, 
    SPEED_CALCULATION, SPEED_LIMITS
)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI app
app = FastAPI(title="Traffic Speed Detection")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình static files và templates
static_dir = Path(__file__).parent.parent / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=os.path.join("src", "web", "templates"))

# Tạo các thư mục cần thiết
os.makedirs(STORAGE_CONFIG['upload_dir'], exist_ok=True)
os.makedirs(STORAGE_CONFIG['processed_dir'], exist_ok=True)
os.makedirs(STORAGE_CONFIG['violations_dir'], exist_ok=True)
os.makedirs(STORAGE_CONFIG['temp_dir'], exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)

# Khởi tạo detector
model = YOLO(STORAGE_CONFIG['model_path'])
speed_detector = SpeedDetector(model)

@app.get("/")
async def read_root():
    """Serve the React app's index.html"""
    index_path = Path(__file__).parent.parent / "web" / "static" / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Index file not found")
    return FileResponse(index_path)

@app.post("/detect")
async def detect_vehicles(file: UploadFile = File(...)):
    """Phát hiện và đo tốc độ phương tiện trong video."""
    if not file.content_type.startswith('video/'):
        logger.error(f"Loại file không hợp lệ: {file.content_type}")
        raise HTTPException(status_code=400, detail="File phải là video")

    try:
        # Lưu video tạm thời
        temp_path = os.path.join(STORAGE_CONFIG['temp_dir'], f"temp_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Xử lý video
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Không thể mở video")

        # Lấy thông tin video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Điều chỉnh độ phân giải nếu cần
        target_width, target_height = CAMERA_CONFIG['resolution']
        if frame_width != target_width or frame_height != target_height:
            logger.info(f"Điều chỉnh độ phân giải video từ {frame_width}x{frame_height} sang {target_width}x{target_height}")
        
        # Tạo đường dẫn cho video đã xử lý
        processed_filename = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        temp_output = os.path.join(STORAGE_CONFIG['processed_dir'], f"temp_{processed_filename}")
        final_output = os.path.join(STORAGE_CONFIG['processed_dir'], processed_filename)
        
        # Khởi tạo VideoWriter với codec mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (target_width, target_height))
        if not out.isOpened():
            raise HTTPException(status_code=500, detail="Không thể tạo video output")

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Điều chỉnh kích thước frame nếu cần
            if frame.shape[1] != target_width or frame.shape[0] != target_height:
                frame = cv2.resize(frame, (target_width, target_height))

            # Điều chỉnh độ sáng và độ tương phản
            frame = cv2.convertScaleAbs(frame, alpha=CAMERA_CONFIG['contrast'], 
                                      beta=CAMERA_CONFIG['exposure'])

            # Phát hiện và tracking đối tượng
            results = speed_detector.process_frame(frame)
            
            # Cập nhật tracking và tính toán tốc độ
            trackers = speed_detector.update(frame, results)
            
            # Lấy thống kê hiện tại
            current_stats = speed_detector.get_statistics()
            
            # Vẽ kết quả
            frame = draw_results(frame, trackers, current_stats)
            
            # Lưu frame
            out.write(frame)
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"Đã xử lý {frame_count} frames")

        cap.release()
        out.release()

        # Chuyển đổi video sang H.264 để tương thích với web
        try:
            ffmpeg_cmd = f'ffmpeg -i {temp_output} -c:v libx264 -preset medium -crf 23 -c:a aac -strict experimental -b:a 128k {final_output}'
            result = os.system(ffmpeg_cmd)
            if result == 0 and os.path.exists(final_output):
                os.remove(temp_output)  # Xóa file tạm
                processed_path = final_output
            else:
                # Nếu chuyển đổi thất bại, sử dụng file mp4v
                os.rename(temp_output, final_output)
                processed_path = final_output
        except Exception as e:
            logger.error(f"Lỗi khi chuyển đổi video: {e}")
            # Sử dụng file mp4v nếu có lỗi
            os.rename(temp_output, final_output)
            processed_path = final_output

        # Xóa file tạm
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Lấy thống kê cuối cùng
        final_stats = speed_detector.get_statistics()

        # Lấy danh sách vi phạm từ database
        conn = sqlite3.connect(STORAGE_CONFIG['database_path'])
        c = conn.cursor()
        c.execute('''
            SELECT timestamp, vehicle_type, speed, speed_limit, image_path, confidence
            FROM violations
            ORDER BY timestamp DESC
            LIMIT 100
        ''')
        violations = [{
            'timestamp': row[0],
            'vehicle_type': row[1],
            'speed': row[2],
            'speed_limit': row[3],
            'image_path': row[4],
            'confidence': row[5]
        } for row in c.fetchall()]
        conn.close()

        # Xóa các file vi phạm cũ (nếu cần)
        current_time = datetime.now()
        for violation in violations:
            try:
                file_time = datetime.strptime(violation['timestamp'], '%Y-%m-%d %H:%M:%S')
                if (current_time - file_time).days > STORAGE_CONFIG['max_storage_days']:
                    if os.path.exists(violation['image_path']):
                        os.remove(violation['image_path'])
            except Exception as e:
                logger.warning(f"Không thể xóa file vi phạm cũ: {e}")

        return JSONResponse(content={
            'status': 'success',
            'message': f'Đã xử lý {frame_count} frames',
            'statistics': {
                'total_vehicles': final_stats['total_vehicles'],
                'active_vehicles': final_stats['active_vehicles'],
                'average_speed': round(final_stats['average_speed'], 2),
                'vehicle_counts': final_stats['vehicle_counts'],
                'violation_count': final_stats['violation_count']
            },
            'violations': violations,
            'processed_video': f'/static/{processed_filename}'
        })

    except Exception as e:
        logger.error(f"Lỗi khi xử lý video: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý video: {str(e)}")

@app.get("/violations")
async def get_violations(limit: int = 100):
    """Lấy danh sách các vi phạm gần đây."""
    try:
        conn = sqlite3.connect(STORAGE_CONFIG['database_path'])
        df = pd.read_sql_query(f'''
            SELECT timestamp, vehicle_type, speed, speed_limit, image_path, confidence
            FROM violations
            ORDER BY timestamp DESC
            LIMIT {limit}
        ''', conn)
        conn.close()
        
        return JSONResponse(content=df.to_dict(orient='records'))
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách vi phạm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics(start_date: str = None, end_date: str = None):
    """Lấy thống kê vi phạm theo khoảng thời gian."""
    try:
        conn = sqlite3.connect(STORAGE_CONFIG['database_path'])
        query = '''
            SELECT 
                date(timestamp) as date,
                COUNT(*) as total_violations,
                AVG(speed) as avg_speed,
                MAX(speed) as max_speed,
                vehicle_type,
                AVG(confidence) as avg_confidence
            FROM violations
        '''
        
        if start_date and end_date:
            query += f" WHERE date(timestamp) BETWEEN '{start_date}' AND '{end_date}'"
            
        query += '''
            GROUP BY date, vehicle_type
            ORDER BY date DESC, total_violations DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return JSONResponse(content={
            'daily_stats': df.to_dict(orient='records'),
            'summary': {
                'total_violations': int(df['total_violations'].sum()),
                'avg_speed': float(df['avg_speed'].mean()),
                'max_speed': float(df['max_speed'].max()),
                'avg_confidence': float(df['avg_confidence'].mean())
            }
        })
    except Exception as e:
        logger.error(f"Lỗi khi lấy thống kê: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/violation/{violation_id}")
async def get_violation_details(request: Request, violation_id: int):
    """Hiển thị chi tiết vi phạm"""
    try:
        conn = sqlite3.connect(STORAGE_CONFIG['database_path'])
        c = conn.cursor()
        c.execute('''
            SELECT timestamp, vehicle_type, speed, speed_limit, image_path, location_x, location_y, confidence
            FROM violations
            WHERE id = ?
        ''', (violation_id,))
        violation = c.fetchone()
        conn.close()
        
        if violation is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy vi phạm")
            
        violation_data = {
            'timestamp': violation[0],
            'vehicle_type': violation[1],
            'speed': violation[2],
            'speed_limit': violation[3],
            'image_path': violation[4],
            'location_x': violation[5],
            'location_y': violation[6],
            'confidence': violation[7]
        }
        
        return templates.TemplateResponse(
            "violation_details.html",
            {
                "request": request,
                "violation": violation_data
            }
        )
    except Exception as e:
        logger.error(f"Lỗi khi lấy chi tiết vi phạm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Handle video upload and processing"""
    try:
        # Save uploaded file
        file_path = static_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process video (implement your video processing logic here)
        detections = []  # Replace with actual detection logic
        
        return {"status": "success", "detections": detections}
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/report")
async def get_report():
    """Get detection report data"""
    try:
        # Implement your report generation logic here
        report_data = []  # Replace with actual report data
        return report_data
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)