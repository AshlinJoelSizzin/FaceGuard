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

<a name="overview"></a>
## Overview

FaceGuard enables users to register their face once and then continuously monitors new uploads for matches against their stored embeddings. When a match exceeding a configured similarity threshold is detected, the system logs the event, associates it with a potential bad actor, and sends a real-time security alert email to the affected user. This allows individuals to detect and respond quickly if their face is used in harmful or unauthorized images.

<a name="key-features"></a>
## Key Features

- User registration with face image and metadata.
- Multi-model face detection (RetinaFace, SCRFD/YOLOv8-face, MTCNN).
- Embedding extraction using InsightFace (ArcFace).
- Vector similarity search using PostgreSQL + pgvector + FAISS.
- Web-based interface for registration, uploads, and viewing detections.
- Detection history and bad-actor tracking (IP, upload count, timestamps).
- Real-time notification via email on suspicious detections.

<a name="system-architecture"></a>
## System Architecture

- The system follows a layered architecture with clear separation between input, processing, feature extraction, matching, storage, and notification.
- **Input Layer**: FastAPI endpoints for registering users, uploading images, fetching detections, and health checks.
- **Processing Layer**: Ensemble of detectors (RetinaFace, SCRFD/YOLO, MTCNN) plus advanced preprocessing (CLAHE, denoising, deduplication, quality checks, bounding box expansion).
- **Feature Extraction Layer**: InsightFace (ArcFace) generates robust 512-D embeddings for detected faces.
- **Matching Layer**: FAISS and pgvector perform fast similarity search with adaptive thresholds and CPU fallback logic.
- **Storage Layer**: PostgreSQL tables for users, user_face_profiles, uploads, detections, and bad_actors.
- **Notification Layer**: SMTP-based email alerts and extendable hooks for SMS/push notifications.
<img width="975" height="623" alt="image" src="https://github.com/user-attachments/assets/119de341-cc88-4c57-8b4b-57f0dbe71fe8" />

<a name="how-it-works"></a>
## How It Works

- 1. A user registers by providing a name, email, password, and a clear face image. The system detects the face, extracts an embedding, and stores it in user_face_profiles with associated metadata.
- 2. When an image is uploaded to the system, YOLOv8-face, SCRFD, RetinaFace, and MTCNN jointly detect faces, and high-quality crops are passed to InsightFace to generate embeddings.
- 3. Each embedding is compared against stored user embeddings using FAISS/pgvector; if similarity exceeds a configurable threshold, a detection record is created and linked to a bad_actor profile based on uploader IP.
- 4. The system sends email alerts to all matched users, including similarity score, detection time, and uploader IP, enabling the said users to review and act if the content is malicious.
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/deaa8ae7-4a49-4408-ac0e-1858f1f78b39" />
<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/808f6f36-8b56-43b2-8eba-a13d17fbde4b" />

<a name="tech-stack"></a>
## Tech Stack

- **Backend**: Python, FastAPI, Hypercorn.
- **Detection**: YOLOv8-face, SCRFD, RetinaFace, MTCNN.
- **Embeddings**: InsightFace (ArcFace, buffalo_l models).
- **Search**: FAISS, PostgreSQL + pgvector extension.
- **Frontend**: HTML templates (Jinja2), basic CSS/JS.
- **Notifications**: SMTP email integration (Gmail or custom SMTP).

<a name="prerequisites"></a>
## Prerequisites

1. Python 3.8 or higher
2. PostgreSQL with pgvector extension installed
3. YOLOv8 face model weights (yolov8x-face-lindevs.onnx)
4. Virtual environment (recommended)

<a name="installation"></a>
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

<a name="running-the-application"></a>
## Running the Application

```bash
python face_monitor_updated.py
```

The application will start on `http://localhost:8000`

<img width="975" height="397" alt="image" src="https://github.com/user-attachments/assets/f5f58466-4498-417d-a523-088623b2aa91" />
<img width="975" height="341" alt="image" src="https://github.com/user-attachments/assets/ae5b19d0-3fca-4e40-92ce-ccdba9c5a514" />


<a name="usage"></a>
## Usage

1. Open your browser and go to `http://localhost:8000`
2. Register a user using the "Register New User" form
3. Upload images using the "Upload Image for Face Detection" form
4. View detections for a user using the "View User Detections" form

<img width="975" height="446" alt="image" src="https://github.com/user-attachments/assets/ce071b07-0c53-4e5e-ae08-d508ec36c676" />
<img width="975" height="334" alt="image" src="https://github.com/user-attachments/assets/7517edd7-8ab9-4d2a-8819-89b471434c7c" />
<img width="948" height="533" alt="image" src="https://github.com/user-attachments/assets/422822a7-e83f-4719-9906-b3203443edde" />
<img width="953" height="298" alt="image" src="https://github.com/user-attachments/assets/624ab143-6be8-4b5f-80c9-8432b5c7132a" />

<a name="api-endpoints"></a>
## API Endpoints

- `GET /` - Web interface
- `POST /register` - Register a new user with face image
- `POST /upload` - Upload an image for face detection
- `GET /user/{user_id}/detections` - Get face detections for a user
- `GET /image?path={path}` - Fetch an image by path

<a name="database-schema"></a>
## Database Schema
- Core tables used by FaceGuard include:
- users – Stores user accounts (id, username, email, password_hash, created_at).
- user_face_profiles – Stores reference face images and corresponding embeddings for each user.
- uploads – Logs image uploads (path, uploader IP, timestamps).
- detections – Records face matches with similarity scores, timestamps, and notification status.
- bad_actors – Tracks uploader IPs, upload_count, last_seen, and created_at for repeated suspicious uploads.

<a name="monitoring-n-alerts"></a>
## Monitoring & Alerts
- When a face in an uploaded image matches a stored profile above the similarity threshold, a detection entry is created and the corresponding user is notified by email.
- The email contains key details: detected user ID, similarity percentage, uploader IP, detection time, and links to review content.
- This allows subscribers to quickly identify unauthorized use of their face in images and take action (e.g., content removal, legal steps).

<img width="975" height="548" alt="image" src="https://github.com/user-attachments/assets/c610ea33-5dbe-49e7-b8e0-f2d670e417f6" />
<img width="1078" height="681" alt="image" src="https://github.com/user-attachments/assets/3f73d64b-a314-4b3a-a40c-914d851180ed" />

<a name="troubleshooting"></a>
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


