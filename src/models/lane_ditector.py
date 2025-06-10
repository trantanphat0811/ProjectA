import cv2
import numpy as np
from data_preparer import VIDEO_CONFIG, LOGGING_CONFIG
import logging

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class LaneDetector:
    def __init__(self):
        self.lane_lines = []  # Lưu các đường làn (dạng [x1, y1, x2, y2])
        self.roi = VIDEO_CONFIG.get('roi', [0, int(0.6*720), 1280, 720])  # Vùng quan tâm (ROI)

    def detect_lanes(self, frame):
        """Phát hiện làn đường sử dụng Hough Transform."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        mask = np.zeros_like(edges)
        roi_corners = np.array([[[self.roi[0], self.roi[1]], [self.roi[0], self.roi[3]],
                                 [self.roi[2], self.roi[3]], [self.roi[2], self.roi[1]]]], dtype=np.int32)
        cv2.fillPoly(mask, roi_corners, 255)
        edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            self.lane_lines = lines[:, 0, :]
            for line in self.lane_lines:
                x1, y1, x2, y2 = line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def check_lane_violation(self, bbox):
        """Kiểm tra vi phạm làn đường."""
        x, y, w, h = map(int, bbox)
        center_x = x + w // 2
        for line in self.lane_lines:
            x1, y1, x2, y2 = line
            if min(y1, y2) > y + h // 2:  # Chỉ kiểm tra dưới bounding box
                dist = abs((y2 - y1) * center_x - (x2 - x1) * (y + h // 2) + x2 * y1 - y2 * x1) / \
                       np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
                if dist < 20:  # Ngưỡng vi phạm (pixel)
                    return True
        return False

if __name__ == "__main__":
    detector = LaneDetector()
    cap = cv2.VideoCapture('sample_video.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.detect_lanes(frame)
        cv2.imshow('Lane Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()