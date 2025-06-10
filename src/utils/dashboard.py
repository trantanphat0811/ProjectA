import streamlit as st
import cv2
import requests
import numpy as np
from PIL import Image
import pandas as pd
import sqlite3
from config import DATABASE_CONFIG, DASHBOARD_CONFIG, MODEL_CONFIG, LOGGING_CONFIG
import logging

# Cấu hình logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Cấu hình Streamlit
st.set_page_config(
    page_title=DASHBOARD_CONFIG['title'],
    page_icon=DASHBOARD_CONFIG['page_icon'],
    layout=DASHBOARD_CONFIG['layout']
)

def get_vehicle_report():
    """Lấy báo cáo từ cơ sở dữ liệu."""
    try:
        conn = sqlite3.connect(DATABASE_CONFIG['path'])
        query = "SELECT class_name, COUNT(*) as count FROM detections GROUP BY class_name"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Lỗi khi lấy báo cáo: {e}")
        return None

def main():
    st.title(DASHBOARD_CONFIG['title'])
    st.sidebar.header(DASHBOARD_CONFIG['sidebar_title'])

    # Upload ảnh hoặc video
    uploaded_file = st.sidebar.file_uploader("Tải lên ảnh hoặc video", type=['jpg', 'png', 'mp4'])
    if uploaded_file:
        try:
            # Gửi file tới API
            files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post('http://localhost:8000/detect', files=files)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Kết quả phát hiện", use_column_width=True)
            else:
                st.error(f"Lỗi từ API: {response.text}")
        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {e}")
            logger.error(f"Lỗi khi xử lý file: {e}")

    # Hiển thị báo cáo
    st.header("Báo cáo thống kê")
    df = get_vehicle_report()
    if df is not None and not df.empty:
        st.dataframe(df)
        st.subheader("Phân bố phương tiện")
        st.bar_chart(df.set_index('class_name')['count'])
    else:
        st.warning("Không có dữ liệu để hiển thị báo cáo")

if __name__ == "__main__":
    main()