# Face Guard - Face Monitoring System

FaceGuard is a face detection and monitoring system that uses YOLOv8-based face detection and InsightFace (ArcFace) embeddings to continuously monitor uploaded images and alert registered users when their face appears in potentially malicious content.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works) 
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Monitoring & Alerts](#monitoring-n-alerts)
- [Troubleshooting](#troubleshooting)

## Overview

FaceGuard enables users to register their face once and then continuously monitors new uploads for matches against their stored embeddings. When a match exceeding a configured similarity threshold is detected, the system logs the event, associates it with a potential bad actor, and sends a real-time security alert email to the affected user. This allows individuals to detect and respond quickly if their face is used in harmful or unauthorized images.

## Key Features

- User registration with face image and metadata.
- Multi-model face detection (RetinaFace, SCRFD/YOLOv8-face, MTCNN).
- Embedding extraction using InsightFace (ArcFace).
- Vector similarity search using PostgreSQL + pgvector + FAISS.
- Web-based interface for registration, uploads, and viewing detections.
- Detection history and bad-actor tracking (IP, upload count, timestamps).
- Real-time notification via email on suspicious detections.

## System Architecture

- The system follows a layered architecture with clear separation between input, processing, feature extraction, matching, storage, and notification.
- **Input Layer**: FastAPI endpoints for registering users, uploading images, fetching detections, and health checks.
- **Processing Layer**: Ensemble of detectors (RetinaFace, SCRFD/YOLO, MTCNN) plus advanced preprocessing (CLAHE, denoising, deduplication, quality checks, bounding box expansion).
- **Feature Extraction Layer**: InsightFace (ArcFace) generates robust 512-D embeddings for detected faces.
- **Matching Layer**: FAISS and pgvector perform fast similarity search with adaptive thresholds and CPU fallback logic.
- **Storage Layer**: PostgreSQL tables for users, user_face_profiles, uploads, detections, and bad_actors.
- **Notification Layer**: SMTP-based email alerts and extendable hooks for SMS/push notifications.

## Prerequisites

1. Python 3.8 or higher
2. PostgreSQL database with pgvector extension
3. Virtual environment (recommended)

<a name="custom-anchor-point"></a>
## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up PostgreSQL database:
   - Install PostgreSQL
   - Create a database named `face_detection_db`
   - Install the pgvector extension (https://github.com/pgvector/pgvector)
   - Update the database credentials in `face_monitor_updated.py` if needed

4. Initialize the database:
   ```bash
   python init_db.py
   ```

5. Download the YOLOv8 face weights:
   - The application expects `yolov8x-face-lindevs.pt` in the root directory
   - You can download it from: https://github.com/akanametov/yolov8-face

## Running the Application

```bash
python face_monitor_updated.py
```

The application will start on `http://localhost:8000`

## Usage

1. Open your browser and go to `http://localhost:8000`
2. Register a user using the "Register New User" form
3. Upload images using the "Upload Image for Face Detection" form
4. View detections for a user using the "View User Detections" form

## API Endpoints

- `GET /` - Web interface
- `POST /register` - Register a new user with face image
- `POST /upload` - Upload an image for face detection
- `GET /user/{user_id}/detections` - Get face detections for a user
- `GET /image?path={path}` - Fetch an image by path

## Troubleshooting

1. If you get database connection errors, make sure:
   - PostgreSQL is running
   - The database `face_detection_db` exists
   - The credentials in `face_monitor_updated.py` are correct
   - The pgvector extension is installed

2. If you get model loading errors, make sure:
   - The `yolov8x-face-lindevs.pt` file is in the root directory
   - All required Python packages are installed
   pt file link:  https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8x-face-lindevs.onnx


