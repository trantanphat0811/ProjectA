from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import shutil
import databases
import sqlalchemy
from pydantic import BaseModel
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI(title="Traffic Speed Detection API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./detections.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# Define database tables
videos = sqlalchemy.Table(
    "videos",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("filename", sqlalchemy.String),
    sqlalchemy.Column("upload_date", sqlalchemy.DateTime),
    sqlalchemy.Column("processed_date", sqlalchemy.DateTime, nullable=True),
    sqlalchemy.Column("total_vehicles", sqlalchemy.Integer, nullable=True),
    sqlalchemy.Column("violations", sqlalchemy.Integer, nullable=True),
)

violations = sqlalchemy.Table(
    "violations",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("video_id", sqlalchemy.Integer),
    sqlalchemy.Column("timestamp", sqlalchemy.DateTime),
    sqlalchemy.Column("speed", sqlalchemy.Float),
    sqlalchemy.Column("vehicle_type", sqlalchemy.String),
    sqlalchemy.Column("image_path", sqlalchemy.String),
)

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Startup event
@app.on_event("startup")
async def startup():
    await database.connect()
    # Create directories if they don't exist
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)

# Shutdown event
@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy"}

# Upload video endpoint
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Save file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Insert video record
        query = videos.insert().values(
            filename=file.filename,
            upload_date=datetime.now()
        )
        video_id = await database.execute(query)
        
        return JSONResponse({
            "message": "File uploaded successfully",
            "video_id": video_id,
            "filename": file.filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get videos endpoint
@app.get("/api/videos")
async def get_videos():
    query = videos.select()
    return await database.fetch_all(query)

# Get violations endpoint
@app.get("/api/violations")
async def get_violations():
    query = violations.select()
    return await database.fetch_all(query)

# Get video details endpoint
@app.get("/api/video/{video_id}")
async def get_video_details(video_id: int):
    # Get video info
    video_query = videos.select().where(videos.c.id == video_id)
    video = await database.fetch_one(video_query)
    
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Get violations for this video
    violations_query = violations.select().where(violations.c.video_id == video_id)
    video_violations = await database.fetch_all(violations_query)
    
    return {
        "video": dict(video),
        "violations": [dict(v) for v in video_violations]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 