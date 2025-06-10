import os
import cv2
import logging.config
import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime

from config import (
    UPLOAD_FOLDER,
    TEMP_FOLDER,
    LOG_CONFIG,
    DATABASE_URL,
    VIOLATION_IMAGE_DIR
)
from speed_detector import SpeedDetector

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'static/processed_videos',
        'static/violations',
        'static/thumbnails',
        'uploads',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

# Configure logging
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)

# Initialize directories
create_directories()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB
app.config['MAX_CONTENT_PATH'] = 500 * 1024 * 1024  # 500MB

# Initialize speed detector
detector = SpeedDetector()

def init_db():
    """Initialize the SQLite database."""
    try:
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        
        # Create violations table
        c.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                speed REAL,
                frame_number INTEGER,
                timestamp TEXT,
                confidence REAL,
                video_file TEXT,
                image_file TEXT
            )
        ''')
        
        # Create processed_videos table
        c.execute('''
            CREATE TABLE IF NOT EXISTS processed_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_filename TEXT,
                filename TEXT,
                processed_date TEXT,
                duration REAL,
                total_vehicles INTEGER,
                total_violations INTEGER,
                avg_speed REAL,
                max_speed REAL,
                min_speed REAL,
                thumbnail TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        logger.error(traceback.format_exc())

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

@app.route('/')
def index():
    """Render the main page."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/processed_videos')
def processed_videos():
    """Display list of processed videos."""
    try:
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        c.execute('SELECT * FROM processed_videos ORDER BY processed_date DESC')
        videos = [{
            'id': row[0],
            'original_filename': row[1],
            'filename': row[2],
            'processed_date': row[3],
            'duration': row[4],
            'total_vehicles': row[5],
            'violations': row[6],
            'avg_speed': row[7],
            'max_speed': row[8],
            'min_speed': row[9],
            'thumbnail': row[10]
        } for row in c.fetchall()]
        conn.close()
        
        return render_template('processed_videos.html', videos=videos)
    except Exception as e:
        logger.error(f"Error displaying processed videos: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/video/<int:video_id>')
def view_video_details(video_id):
    """Display details of a specific processed video."""
    try:
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        
        # Get video details
        c.execute('SELECT * FROM processed_videos WHERE id = ?', (video_id,))
        video_row = c.fetchone()
        if not video_row:
            return jsonify({'error': 'Video not found'}), 404
            
        video = {
            'id': video_row[0],
            'original_filename': video_row[1],
            'filename': video_row[2],
            'processed_date': video_row[3],
            'duration': video_row[4],
            'total_vehicles': video_row[5],
            'violations': video_row[6],
            'avg_speed': video_row[7],
            'max_speed': video_row[8],
            'min_speed': video_row[9],
            'thumbnail': video_row[10]
        }
        
        # Get violations for this video
        c.execute('SELECT * FROM violations WHERE video_file = ? ORDER BY frame_number', (video['filename'],))
        violations = [{
            'id': row[0],
            'track_id': row[1],
            'speed': row[2],
            'frame_number': row[3],
            'timestamp': row[4],
            'confidence': row[5],
            'video_file': row[6],
            'image_file': row[7]
        } for row in c.fetchall()]
        
        conn.close()
        return render_template('video_details.html', video=video, violations=violations)
    except Exception as e:
        logger.error(f"Error displaying video details: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle video file upload and processing."""
    try:
        logger.info("Starting file upload process")
        if 'video' not in request.files:
            logger.warning("No video file in request")
            return jsonify({'error': 'No video file provided'}), 400
            
        file = request.files['video']
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to {filepath}")
        file.save(filepath)
        
        # Process video
        logger.info("Starting video processing")
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {filepath}")
            return jsonify({'error': 'Could not open video file'}), 500

        # Get video information
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        frame_count = 0
        violations = []
        total_vehicles = 0
        speeds = []
        
        # Create processed video file
        processed_filename = f"processed_{filename}"
        processed_filepath = os.path.join('static/processed_videos', processed_filename)
        
        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(processed_filepath, fourcc, fps, (width, height))
        
        # Create thumbnail
        ret, thumbnail_frame = cap.read()
        if ret:
            thumbnail_filename = f"thumb_{filename.rsplit('.', 1)[0]}.jpg"
            thumbnail_path = os.path.join('static/thumbnails', thumbnail_filename)
            cv2.imwrite(thumbnail_path, thumbnail_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        
        current_stats = {
            'vehicles': 0,
            'violations': 0,
            'avg_speed': 0
        }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Process frame
                logger.debug(f"Processing frame {frame_count}")
                annotated_frame, detections = detector.process_frame(frame, frame_count)
                
                # Update current statistics
                if detections:
                    current_stats['vehicles'] = len(set(d['track_id'] for d in detections))
                    current_stats['violations'] = len([d for d in detections if d['is_violation']])
                    speeds_list = [d['speed'] for d in detections if d['speed'] is not None]
                    if speeds_list:
                        current_stats['avg_speed'] = sum(speeds_list) / len(speeds_list)
                
                # Write processed frame
                out.write(annotated_frame)
                
                # Save violations to database
                for detection in detections:
                    if detection.get('is_violation'):
                        violation_img_filename = f"violation_{filename}_{frame_count}_{detection['track_id']}.jpg"
                        violation_img_path = os.path.join('static/violations', violation_img_filename)
                        cv2.imwrite(violation_img_path, annotated_frame)
                        
                        conn = sqlite3.connect('detections.db')
                        c = conn.cursor()
                        c.execute('''
                            INSERT INTO violations 
                            (track_id, speed, frame_number, timestamp, confidence, video_file, image_file)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            detection['track_id'],
                            detection['speed'],
                            frame_count,
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            detection['confidence'],
                            processed_filename,
                            violation_img_filename
                        ))
                        conn.commit()
                        conn.close()
                        violations.append(detection)
                        speeds.append(detection['speed'])
                
                frame_count += 1
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        cap.release()
        out.release()
        
        # Calculate final statistics
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        min_speed = min(speeds) if speeds else 0
        
        # Save video metadata to database
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO processed_videos 
            (original_filename, filename, processed_date, duration, total_vehicles, 
             total_violations, avg_speed, max_speed, min_speed, thumbnail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            processed_filename,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            duration,
            current_stats['vehicles'],
            len(violations),
            avg_speed,
            max_speed,
            min_speed,
            thumbnail_filename
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Video processing completed: {filename}")
        return jsonify({
            'message': 'Video processed successfully',
            'stats': {
                'total_frames': total_frames,
                'processed_frames': frame_count,
                'total_vehicles': current_stats['vehicles'],
                'total_violations': current_stats['violations'],
                'avg_speed': current_stats['avg_speed']
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/violations')
def violations_page():
    """Display all violations."""
    try:
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        
        # Get all violations with video information
        c.execute('''
            SELECT v.*, pv.original_filename 
            FROM violations v
            JOIN processed_videos pv ON v.video_file = pv.filename
            ORDER BY v.timestamp DESC
        ''')
        
        violations = [{
            'id': row[0],
            'track_id': row[1],
            'speed': row[2],
            'frame_number': row[3],
            'timestamp': row[4],
            'confidence': row[5],
            'video_file': row[6],
            'image_file': row[7],
            'original_video': row[8]
        } for row in c.fetchall()]
        
        # Get statistics
        c.execute('SELECT COUNT(*) FROM violations')
        total_violations = c.fetchone()[0]
        
        c.execute('SELECT AVG(speed) FROM violations')
        avg_speed = c.fetchone()[0] or 0
        
        c.execute('SELECT COUNT(DISTINCT video_file) FROM violations')
        total_videos = c.fetchone()[0]
        
        conn.close()
        
        return render_template('violations.html', 
                             violations=violations,
                             total_violations=total_violations,
                             avg_speed=round(avg_speed, 1),
                             total_videos=total_videos)
    except Exception as e:
        logger.error(f"Error displaying violations page: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/static/violations/<filename>')
def violation_image(filename):
    """Serve violation images."""
    try:
        return send_from_directory(VIOLATION_IMAGE_DIR, filename)
    except Exception as e:
        logger.error(f"Error serving violation image {filename}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get dashboard statistics."""
    try:
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        
        # Get total vehicles (unique track_ids)
        c.execute('SELECT COUNT(DISTINCT track_id) FROM violations')
        total_vehicles = c.fetchone()[0]
        
        # Get total violations
        c.execute('SELECT COUNT(*) FROM violations')
        total_violations = c.fetchone()[0]
        
        # Get total processed videos
        c.execute('SELECT COUNT(*) FROM processed_videos')
        total_videos = c.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_vehicles': total_vehicles,
            'total_violations': total_violations,
            'total_videos': total_videos
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.errorhandler(500)
def handle_500_error(e):
    logger.error(f"Internal server error: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({'error': 'Internal server error'}), 500

# Initialize application
if __name__ == '__main__':
    create_directories()
    init_db()
    app.run(debug=True) 