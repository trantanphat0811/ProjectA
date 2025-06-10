import cv2
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from src.core.config import CAMERA_CONFIG, STORAGE_CONFIG
from src.utils.visualization import draw_results

def process_video(video_path, speed_detector, output_path=None, progress_callback=None):
    """Process video and detect speeds
    
    Args:
        video_path: Path to input video
        speed_detector: SpeedDetector instance
        output_path: Optional path to save processed video
        progress_callback: Optional callback function to report progress
        
    Returns:
        dict: Processing statistics
    """
    try:
        # Initialize video capture
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")
            
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if needed
        out = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, 
                                (CAMERA_CONFIG['width'], CAMERA_CONFIG['height']))
            
        frame_count = 0
        start_time = datetime.now()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            frame = cv2.resize(frame, (CAMERA_CONFIG['width'], CAMERA_CONFIG['height']))
            frame = cv2.convertScaleAbs(frame, 
                                      alpha=CAMERA_CONFIG['contrast'],
                                      beta=CAMERA_CONFIG['exposure'])
            
            # Process frame
            detections = speed_detector.process_frame(frame)
            
            # Get current statistics
            stats = speed_detector.get_statistics()
            
            # Draw results
            frame = draw_results(frame, detections, stats)
            
            # Save frame if needed
            if out:
                out.write(frame)
                
            # Update progress
            frame_count += 1
            if progress_callback and frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)
                
            if frame_count % 100 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_real = frame_count / elapsed
                logging.info(f"Processed {frame_count}/{total_frames} frames ({fps_real:.1f} fps)")
                
        # Release resources
        cap.release()
        if out:
            out.write(frame)
            out.release()
            
        # Calculate final statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        avg_fps = frame_count / processing_time
        
        final_stats = {
            'frames_processed': frame_count,
            'processing_time': processing_time,
            'average_fps': avg_fps,
            **speed_detector.get_statistics()
        }
        
        logging.info(f"Video processing completed: {final_stats}")
        return final_stats
        
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise
        
def extract_frames(video_path, output_dir, interval=1):
    """Extract frames from video at specified interval
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        interval: Extract every nth frame
        
    Returns:
        int: Number of frames extracted
        """
    try:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")
            
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1
                
            frame_count += 1
            
        cap.release()
        logging.info(f"Extracted {saved_count} frames to {output_dir}")
        return saved_count
        
    except Exception as e:
        logging.error(f"Error extracting frames: {str(e)}")
        raise