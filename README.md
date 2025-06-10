# Traffic Speed Detection System

Hệ thống phát hiện tốc độ phương tiện giao thông sử dụng YOLOv8 và OpenCV.

## Tính năng

- Phát hiện và theo dõi phương tiện trong video
- Tính toán tốc độ của phương tiện
- Ghi nhận các vi phạm tốc độ
- Giao diện web thân thiện với người dùng
- Thống kê và biểu đồ trực quan
- Quản lý video và vi phạm

## Yêu cầu hệ thống

- Python 3.8+
- Node.js 14+
- npm 6+

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd traffic-speed-detection
```

2. Cài đặt dependencies cho backend:
```bash
cd server
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Cài đặt dependencies cho frontend:
```bash
cd ../client
npm install
```

## Cấu hình

1. Backend (server/app.py):
- Port mặc định: 5000
- Upload directory: server/uploads
- Database: SQLite (detections.db)

2. Frontend (client):
- Port mặc định: 3000
- Proxy đến backend: http://localhost:5000

## Chạy ứng dụng

1. Khởi động backend:
```bash
cd server
source venv/bin/activate  # Windows: venv\Scripts\activate
python app.py
```

2. Khởi động frontend (trong terminal mới):
```bash
cd client
npm start
```

3. Truy cập ứng dụng:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## Sử dụng

1. Upload video:
- Truy cập trang chủ
- Click nút "Upload Video"
- Chọn file video (.mp4, .avi, .mov)
- Chờ xử lý

2. Xem kết quả:
- Danh sách video đã xử lý
- Chi tiết vi phạm
- Thống kê và biểu đồ
- Tải xuống báo cáo

## Cấu trúc dự án

```
traffic-speed-detection/
├── client/                 # Frontend React
│   ├── public/
│   ├── src/
│   └── package.json
├── server/                 # Backend Flask
│   ├── api/
│   ├── utils/
│   ├── models/
│   ├── static/
│   ├── uploads/
│   └── app.py
├── requirements.txt
└── README.md
```

## Đóng góp

Vui lòng đọc [CONTRIBUTING.md](CONTRIBUTING.md) để biết thêm chi tiết về quy trình đóng góp.

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.