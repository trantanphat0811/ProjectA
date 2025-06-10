from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
import json
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database setup
def get_db_connection():
    DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///detections.db')
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    
    if DATABASE_URL.startswith('sqlite:///'):
        return sqlite3.connect(DATABASE_URL[10:])
    else:
        import psycopg2
        return psycopg2.connect(DATABASE_URL)

# Routes
@app.route('/')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename
        }), 200
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/videos', methods=['GET'])
def get_videos():
    conn = get_db_connection()
    videos = conn.execute('SELECT * FROM videos ORDER BY upload_date DESC').fetchall()
    conn.close()
    return jsonify([dict(video) for video in videos])

@app.route('/api/violations', methods=['GET'])
def get_violations():
    conn = get_db_connection()
    violations = conn.execute('SELECT * FROM violations ORDER BY timestamp DESC').fetchall()
    conn.close()
    return jsonify([dict(violation) for violation in violations])

@app.route('/api/video/<video_id>', methods=['GET'])
def get_video_details(video_id):
    conn = get_db_connection()
    video = conn.execute('SELECT * FROM videos WHERE id = ?', (video_id,)).fetchone()
    violations = conn.execute('SELECT * FROM violations WHERE video_id = ?', (video_id,)).fetchall()
    conn.close()
    
    if video is None:
        return jsonify({'error': 'Video not found'}), 404
        
    return jsonify({
        'video': dict(video),
        'violations': [dict(v) for v in violations]
    })

@app.route('/api/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 