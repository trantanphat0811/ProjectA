"""
Traffic Speed Detection System
A system for detecting and tracking vehicles, calculating their speeds,
and identifying speed limit violations using YOLOv8 and computer vision.
"""

from .app import app
from .speed_detector import SpeedDetector
from .config import *

__version__ = '1.0.0'
