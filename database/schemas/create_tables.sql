-- Create tables for traffic speed detection system

-- Vehicle detections table
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY,
    timestamp TEXT,
    vehicle_type TEXT,
    confidence REAL,
    speed REAL,
    frame_number INTEGER,
    video_path TEXT,
    bbox_coordinates TEXT,
    track_id INTEGER
);

-- Speed violations table
CREATE TABLE violations (
    violation_id INTEGER PRIMARY KEY,
    detection_id INTEGER,
    timestamp TEXT,
    speed_limit REAL,
    measured_speed REAL,
    vehicle_type TEXT,
    evidence_path TEXT,
    status TEXT
);

-- System configurations table
CREATE TABLE configurations (
    config_id INTEGER PRIMARY KEY,
    parameter_name TEXT,
    parameter_value TEXT,
    description TEXT,
    last_updated TEXT
);

-- Analytics table
CREATE TABLE analytics (
    analytics_id INTEGER PRIMARY KEY,
    date TEXT,
    hour INTEGER,
    total_vehicles INTEGER,
    total_violations INTEGER,
    average_speed REAL,
    peak_hour INTEGER,
    notes TEXT
);

-- Camera calibration table
CREATE TABLE camera_calibration (
    calibration_id INTEGER PRIMARY KEY,
    camera_location TEXT,
    perspective_matrix TEXT,
    reference_points TEXT,
    calibration_date TEXT,
    is_active INTEGER
);

-- Initial configuration values
INSERT INTO configurations (parameter_name, parameter_value, description) VALUES
    ('speed_limit', '50', 'Speed limit in km/h'),
    ('detection_threshold', '0.5', 'Minimum confidence threshold for detection'),
    ('tracking_persistence', '30', 'Number of frames to keep tracking an object'),
    ('violation_clip_duration', '5', 'Duration of violation video clips in seconds'); 