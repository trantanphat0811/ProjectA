"""
Initial database migration script.
Creates the base tables for the traffic speed detection system.
"""

import sqlite3
import logging
from pathlib import Path

def upgrade(db_path):
    """
    Upgrade database to version 1.
    Creates initial tables and configuration.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Read and execute schema
        schema_path = Path(__file__).parent.parent / "schemas" / "create_tables.sql"
        with open(schema_path, 'r') as f:
            cursor.executescript(f.read())
        
        # Record migration version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("INSERT INTO migrations (version) VALUES (?)", (1,))
        
        conn.commit()
        logging.info("Successfully applied migration version 1")
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error applying migration: {e}")
        raise
    finally:
        conn.close()

def downgrade(db_path):
    """
    Downgrade database from version 1.
    Removes all tables.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Drop all tables
        tables = [
            'migrations',
            'detections',
            'violations',
            'configurations',
            'analytics',
            'camera_calibration'
        ]
        
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        
        conn.commit()
        logging.info("Successfully downgraded from version 1")
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error downgrading migration: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db_path = Path(__file__).parent.parent.parent / "detections.db"
    upgrade(str(db_path)) 