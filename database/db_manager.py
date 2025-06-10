import sqlite3
import json
from datetime import datetime
import logging
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path="detections.db"):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.setup_logging()
        self.initialize_database()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/database.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DatabaseManager')

    def connect(self):
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            self.logger.error(f"Database connection error: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def initialize_database(self):
        """Initialize database with schema."""
        try:
            self.connect()
            schema_path = Path(__file__).parent / "schemas" / "create_tables.sql"
            with open(schema_path, 'r') as f:
                self.cursor.executescript(f.read())
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            raise
        finally:
            self.close()

    def add_detection(self, vehicle_type, confidence, speed, frame_number, video_path, bbox_coordinates, track_id):
        """Add a new vehicle detection."""
        try:
            self.connect()
            query = """
                INSERT INTO detections 
                (vehicle_type, confidence, speed, frame_number, video_path, bbox_coordinates, track_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            bbox_json = json.dumps(bbox_coordinates)
            self.cursor.execute(query, (vehicle_type, confidence, speed, frame_number, 
                                      video_path, bbox_json, track_id))
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error adding detection: {e}")
            raise
        finally:
            self.close()

    def add_violation(self, detection_id, speed_limit, measured_speed, vehicle_type, evidence_path):
        """Record a speed violation."""
        try:
            self.connect()
            query = """
                INSERT INTO violations 
                (detection_id, speed_limit, measured_speed, vehicle_type, evidence_path)
                VALUES (?, ?, ?, ?, ?)
            """
            self.cursor.execute(query, (detection_id, speed_limit, measured_speed, 
                                      vehicle_type, evidence_path))
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error adding violation: {e}")
            raise
        finally:
            self.close()

    def get_config_value(self, parameter_name):
        """Retrieve configuration value."""
        try:
            self.connect()
            query = "SELECT parameter_value FROM configurations WHERE parameter_name = ?"
            self.cursor.execute(query, (parameter_name,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            self.logger.error(f"Error getting configuration: {e}")
            raise
        finally:
            self.close()

    def update_config(self, parameter_name, parameter_value):
        """Update configuration parameter."""
        try:
            self.connect()
            query = """
                UPDATE configurations 
                SET parameter_value = ?, last_updated = CURRENT_TIMESTAMP
                WHERE parameter_name = ?
            """
            self.cursor.execute(query, (parameter_value, parameter_name))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            raise
        finally:
            self.close()

    def add_analytics(self, date, hour, total_vehicles, total_violations, average_speed, peak_hour=False, notes=None):
        """Add analytics data."""
        try:
            self.connect()
            query = """
                INSERT INTO analytics 
                (date, hour, total_vehicles, total_violations, average_speed, peak_hour, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            self.cursor.execute(query, (date, hour, total_vehicles, total_violations, 
                                      average_speed, peak_hour, notes))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error adding analytics: {e}")
            raise
        finally:
            self.close()

    def get_violations(self, start_date=None, end_date=None, status=None):
        """Retrieve violations with optional filtering."""
        try:
            self.connect()
            query = "SELECT * FROM violations WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            if status:
                query += " AND status = ?"
                params.append(status)

            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            self.logger.error(f"Error retrieving violations: {e}")
            raise
        finally:
            self.close()

    def update_violation_status(self, violation_id, status):
        """Update violation status."""
        try:
            self.connect()
            query = "UPDATE violations SET status = ? WHERE violation_id = ?"
            self.cursor.execute(query, (status, violation_id))
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error updating violation status: {e}")
            raise
        finally:
            self.close()

    def get_daily_statistics(self, date):
        """Get statistics for a specific date."""
        try:
            self.connect()
            query = """
                SELECT 
                    COUNT(*) as total_detections,
                    AVG(speed) as avg_speed,
                    MAX(speed) as max_speed,
                    COUNT(CASE WHEN speed > ? THEN 1 END) as violations
                FROM detections 
                WHERE DATE(timestamp) = ?
            """
            speed_limit = float(self.get_config_value('speed_limit'))
            self.cursor.execute(query, (speed_limit, date))
            return self.cursor.fetchone()
        except Exception as e:
            self.logger.error(f"Error getting daily statistics: {e}")
            raise
        finally:
            self.close() 