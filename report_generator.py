import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_preparer import DATABASE_CONFIG, OUTPUT_DIR, LOGGING_CONFIG  # Sửa import
import logging

# Cấu hình logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, db_path=DATABASE_CONFIG['path']):
        self.db_path = db_path

    def generate_summary_report(self, output_file=None):
        """Tạo báo cáo thống kê số lượng phương tiện theo loại."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT class_name, COUNT(*) as count FROM detections GROUP BY class_name"
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                logger.warning("Không có dữ liệu phát hiện để tạo báo cáo")
                return None

            # Lưu báo cáo CSV
            if output_file:
                output_path = Path(OUTPUT_DIR) / output_file
                df.to_csv(output_path, index=False)
                logger.info(f"Báo cáo đã được lưu vào {output_path}")

            # In báo cáo
            print("\nBáo cáo thống kê phương tiện:")
            print(df)
            return df

        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo: {e}")
            return None

    def plot_vehicle_distribution(self, save_path=None):
        """Vẽ biểu đồ phân bố số lượng phương tiện theo loại."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT class_name, COUNT(*) as count FROM detections GROUP BY class_name"
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                logger.warning("Không có dữ liệu để vẽ biểu đồ")
                return

            plt.figure(figsize=(10, 6))
            sns.barplot(x='class_name', y='count', data=df)
            plt.title('Phân bố số lượng phương tiện')
            plt.xlabel('Loại phương tiện')
            plt.ylabel('Số lượng')
            plt.xticks(rotation=45)

            if save_path:
                output_path = Path(OUTPUT_DIR) / save_path
                plt.savefig(output_path)
                logger.info(f"Biểu đồ đã được lưu vào {output_path}")
            plt.show()

        except Exception as e:
            logger.error(f"Lỗi khi vẽ biểu đồ: {e}")

if __name__ == "__main__":
    reporter = ReportGenerator()
    reporter.generate_summary_report('vehicle_summary.csv')
    reporter.plot_vehicle_distribution('vehicle_distribution.png')