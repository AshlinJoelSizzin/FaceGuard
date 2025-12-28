# Face Guard - Face Monitoring System

A face detection and monitoring system using InsightFace and YOLOv8 for facial recognition and monitoring.
[Jump to the specific text](#custom-anchor-point)

## Features

- User registration with face image
- Face detection in uploaded images
- Match detection against registered users
- Web interface for easy interaction
- PostgreSQL database with vector similarity search

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


