# Traffic Speed Detection System Database

This directory contains the database structure and management code for the traffic speed detection system.

## Directory Structure

```
database/
├── schemas/           # SQL schema definitions
├── migrations/        # Database migration scripts
├── db_manager.py     # Python database interface
└── README.md         # This file
```

## Database Schema

The system uses SQLite3 and consists of the following tables:

### 1. detections
Stores all vehicle detections from the video feed
- `detection_id`: Primary key
- `timestamp`: Detection time
- `vehicle_type`: Type of vehicle detected
- `confidence`: Detection confidence score
- `speed`: Calculated speed
- `frame_number`: Video frame number
- `video_path`: Source video file path
- `bbox_coordinates`: Bounding box coordinates (JSON)
- `track_id`: Vehicle tracking ID

### 2. violations
Records speed limit violations
- `violation_id`: Primary key
- `detection_id`: Reference to detection
- `timestamp`: Violation time
- `speed_limit`: Speed limit at time of violation
- `measured_speed`: Recorded vehicle speed
- `vehicle_type`: Type of vehicle
- `evidence_path`: Path to violation evidence
- `status`: Violation status (pending/reviewed/archived)

### 3. configurations
System configuration parameters
- `config_id`: Primary key
- `parameter_name`: Configuration parameter name
- `parameter_value`: Parameter value
- `description`: Parameter description
- `last_updated`: Last update timestamp

### 4. analytics
Traffic analytics data
- `analytics_id`: Primary key
- `date`: Date of analytics
- `hour`: Hour of day
- `total_vehicles`: Total vehicles detected
- `total_violations`: Total violations
- `average_speed`: Average speed
- `peak_hour`: Peak hour flag
- `notes`: Additional notes

### 5. camera_calibration
Camera calibration data
- `calibration_id`: Primary key
- `camera_location`: Camera location description
- `perspective_matrix`: Perspective transformation matrix
- `reference_points`: Calibration reference points
- `calibration_date`: Calibration timestamp
- `is_active`: Active calibration flag

## Usage

### Database Manager

The `DatabaseManager` class in `db_manager.py` provides a high-level interface for database operations:

```python
from database.db_manager import DatabaseManager

# Initialize database
db = DatabaseManager()

# Add detection
detection_id = db.add_detection(
    vehicle_type="car",
    confidence=0.95,
    speed=45.5,
    frame_number=1234,
    video_path="videos/traffic.mp4",
    bbox_coordinates={"x1": 100, "y1": 200, "x2": 300, "y2": 400},
    track_id=1
)

# Record violation
violation_id = db.add_violation(
    detection_id=detection_id,
    speed_limit=40.0,
    measured_speed=45.5,
    vehicle_type="car",
    evidence_path="evidence/violation_001.jpg"
)

# Get configuration
speed_limit = db.get_config_value("speed_limit")

# Update configuration
db.update_config("detection_threshold", "0.6")

# Get daily statistics
stats = db.get_daily_statistics("2024-03-15")
```

### Database Migrations

To apply database migrations:

```bash
python -m database.migrations.001_initial_schema
```

To create a new migration:
1. Create a new file in `migrations/` with sequential numbering
2. Implement `upgrade()` and `downgrade()` functions
3. Run the migration script

## Maintenance

- Regular backups are recommended
- Monitor database size and performance
- Review and archive old violations periodically
- Update configurations as needed for optimal performance 