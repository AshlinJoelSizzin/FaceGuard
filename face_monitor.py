import os
import shutil
import numpy as np
import cv2
import asyncpg
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import albumentations as A  
import hashlib
import logging
import json
from PIL import Image
# from huggingface_hub import hf_hub_download


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:AshlinJoelSizzn@localhost:5432/face_detection_db"  # <-- PUT your password!
USER_IMG_DIR = "user_images"
ACTOR_IMG_DIR = "actor_uploads"
YOLOV8_FACE_WEIGHTS = "yolov8x-face-lindevs.pt"

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Create directories
os.makedirs(USER_IMG_DIR, exist_ok=True)
os.makedirs(ACTOR_IMG_DIR, exist_ok=True)

# --- ML MODELS ---
print("Loading InsightFace models...")
try:
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    logger.error(f"Failed to load InsightFace: {e}")
    face_app = None

print("Loading YOLOv8-face model...")
try:
    if os.path.exists(YOLOV8_FACE_WEIGHTS):
        yolo_model = YOLO(YOLOV8_FACE_WEIGHTS)
    else:
        logger.warning(f"YOLO weights file not found: {YOLOV8_FACE_WEIGHTS}")
        yolo_model = None
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    yolo_model = None

AUGMENTER = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(p=0.2),
    A.RandomGamma(p=0.4),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=12, p=0.4)
])

async def get_db():
    """Get database connection with error handling"""
    try:
        return await asyncpg.connect(DB_URL)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(500, "Database connection failed")

def hash_img(path: str) -> str:
    """Generate hash of image file"""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        logger.error(f"Failed to hash image {path}: {e}")
        return ""

def save_upload(upload_dir, filename, file):
    """Save uploaded file with error handling"""
    try:
        os.makedirs(upload_dir, exist_ok=True)
        dest = os.path.join(upload_dir, filename)
        with open(dest, "wb") as buf:
            shutil.copyfileobj(file, buf)
        return dest
    except Exception as e:
        logger.error(f"Failed to save file {filename}: {e}")
        raise HTTPException(500, f"Failed to save file: {str(e)}")

def iou_bbox(a, b):
    """Calculate Intersection over Union for two bounding boxes"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    
    # Calculate intersection
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    
    # Check if there's actual intersection
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    
    union_area = a_area + b_area - inter_area
    return inter_area / float(union_area + 1e-6) if union_area > 0 else 0.0

def ensemble_detect_faces(image: np.ndarray):
    """Detect faces using ensemble of SCRFD and YOLOv8"""
    if image is None or image.size == 0:
        return []
    
    detections = []
    
    # SCRFD detection
    faces_scrfd = []
    if face_app is not None:
        try:
            faces_scrfd = face_app.get(image)
        except Exception as e:
            logger.error(f"SCRFD detection failed: {e}")
    
    # YOLOv8 detection
    boxes_yolo = []
    if yolo_model is not None:
        try:
            yolo_results = yolo_model.predict(source=image, conf=0.5, verbose=False)
            for result in yolo_results:
                if result.boxes is not None:
                    for box in result.boxes:
                        coords = box.xyxy.flatten()
                        if len(coords) >= 4:
                            x1, y1, x2, y2 = map(int, coords[:4])
                            # Validate coordinates
                            if x2 > x1 and y2 > y1:
                                boxes_yolo.append((x1, y1, x2, y2))
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
    
    # Process SCRFD detections
    for face in faces_scrfd:
        try:
            bbox = face.get("bbox", [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                # Validate bbox
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                    overlap = any(iou_bbox((x1,y1,x2,y2), yb) > 0.4 for yb in boxes_yolo)
                    if overlap or not boxes_yolo:
                        embedding = face.get("embedding", [])
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.tolist()
                        
                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "aligned": face.get("aligned"),
                            "embedding": embedding,
                            "score": float(face.get("det_score", 0.0))
                        })
        except Exception as e:
            logger.error(f"Error processing SCRFD face: {e}")
            continue
    
    # Add YOLO-only detections
    for bx in boxes_yolo:
        try:
            if not any(iou_bbox(bx, (int(f["bbox"][0]), int(f["bbox"][1]), 
                                   int(f["bbox"][2]), int(f["bbox"][3]))) > 0.4 
                      for f in faces_scrfd if len(f.get("bbox", [])) >= 4):
                
                x1, y1, x2, y2 = bx
                # Validate crop dimensions
                if (y2 > y1 and x2 > x1 and y1 >= 0 and x1 >= 0 and 
                    y2 <= image.shape[0] and x2 <= image.shape[1]):
                    
                    crop = image[y1:y2, x1:x2]
                    if crop is not None and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                        crop_resized = cv2.resize(crop, (112, 112))
                        
                        if face_app is not None:
                            try:
                                faces = face_app.get(crop_resized)
                                if faces:
                                    embedding = faces[0].get("embedding", [])
                                    if isinstance(embedding, np.ndarray):
                                        embedding = embedding.tolist()
                                    
                                    detections.append({
                                        "bbox": bx,
                                        "aligned": crop_resized,
                                        "embedding": embedding,
                                        "score": 0.5
                                    })
                            except Exception as e:
                                logger.error(f"Failed to get embedding for YOLO detection: {e}")
        except Exception as e:
            logger.error(f"Error processing YOLO detection: {e}")
            continue
    
    return detections

def augment_face(face_img: np.ndarray, n=4):
    """Generate augmented versions of face image"""
    if face_img is None or face_img.size == 0:
        return []
    
    augmented = []
    for _ in range(n):
        try:
            aug_img = AUGMENTER(image=face_img)["image"]
            augmented.append(aug_img)
        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            continue
    return augmented

def cosine_sim(a, b):
    """Calculate cosine similarity between two vectors"""
    try:
        a_np = np.array(a) if not isinstance(a, np.ndarray) else a
        b_np = np.array(b) if not isinstance(b, np.ndarray) else b

        # Handle empty or invalid arrays
        if a_np.size == 0 or b_np.size == 0:
            return 0.0

        # Normalize embeddings for better similarity calculation
        a_normalized = a_np / (np.linalg.norm(a_np) + 1e-8)
        b_normalized = b_np / (np.linalg.norm(b_np) + 1e-8)
        
        # Calculate cosine similarity (dot product of normalized vectors)
        similarity = float(np.dot(a_normalized, b_normalized))
        
        return similarity

    except Exception as e:
        logger.error(f"Cosine similarity calculation failed: {e}")
        return 0.0


app = FastAPI()

@app.post("/register")
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    image: UploadFile = File(...)
):
    """Register a new user with face profile"""
    conn = None
    img_path = None

    try:
        # Check if models are loaded
        if face_app is None:
            raise HTTPException(500, "Face recognition model not available")
        
        # Save uploaded image
        img_path = save_upload(USER_IMG_DIR, f"{username}_{image.filename}", image.file)
        
        # Read and validate image
        pil_img = Image.open(img_path).convert('RGB')
        img_np = np.array(pil_img)
        img= cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if img is None:
            raise HTTPException(400, "Could not read image. Invalid or unsupported format.")
        
        # Detect faces using ensemble method
        faces_found = ensemble_detect_faces(img)
        if not faces_found:
            raise HTTPException(400, "No face detected in image.")
        
        logger.info(f"Found {len(faces_found)} faces")
        
        # Generate embeddings - USE EMBEDDINGS DIRECTLY FROM DETECTION
        embeddings = []
        
        for i, face in enumerate(faces_found):
            try:
                # Get embedding directly from face detection result
                base_embedding = face.get("embedding", [])
                logger.info(f"Face {i}: embedding type = {type(base_embedding)}")
                
                if base_embedding is not None:
                    # Convert to list if numpy array
                    if isinstance(base_embedding, np.ndarray):
                        base_emb_list = base_embedding.tolist()
                    elif isinstance(base_embedding, list):
                        base_emb_list = base_embedding
                    else:
                        logger.warning(f"Face {i}: Unknown embedding type {type(base_embedding)}")
                        continue
                    
                    # Validate embedding is not empty
                    if base_emb_list and len(base_emb_list) > 0:
                        embeddings.append(base_emb_list)
                        logger.info(f"Face {i}: Added embedding of length {len(base_emb_list)}")
                        
                        # OPTIONAL: Add augmented versions by slightly modifying the original
                        # (Instead of re-running face detection on augmented images)
                        aligned_face = face.get("aligned")
                        if aligned_face is not None:
                            try:
                                # Generate a few augmented versions
                                aug_faces = augment_face(aligned_face, n=2)  # Reduce from 4 to 2
                                for aug_face in aug_faces:
                                    if aug_face is not None:
                                        # Try to get embedding from augmented face
                                        aug_results = face_app.get(aug_face)
                                        if aug_results and len(aug_results) > 0:
                                            aug_emb = aug_results[0].get("embedding", [])
                                            if isinstance(aug_emb, np.ndarray):
                                                embeddings.append(aug_emb.tolist())
                                            elif isinstance(aug_emb, list) and len(aug_emb) > 0:
                                                embeddings.append(aug_emb)
                            except Exception as e:
                                logger.warning(f"Augmentation failed for face {i}: {e}")
                                # Continue with just the base embedding
                    else:
                        logger.warning(f"Face {i}: Empty embedding")
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")
                continue
        
        logger.info(f"Total embeddings generated: {len(embeddings)}")
        
        if not embeddings:
            raise HTTPException(400, "Failed to generate face embeddings")
        
        # Database operations
        conn = await get_db()
        
        # Check if username already exists
        existing_user = await conn.fetchval("SELECT user_id FROM users WHERE username=$1", username)
        if existing_user:
            raise HTTPException(400, "Username already exists")
        
        # Insert user
        await conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3)",
            username, email, password  # In production, hash the password!
        )
        
        user_id = await conn.fetchval("SELECT user_id FROM users WHERE username=$1", username)
        if user_id is None:
            raise HTTPException(500, "Failed to create user")
        
        # Insert face profile
        await conn.execute(
        "INSERT INTO user_face_profiles (user_id, image_path, embeddings, augment_count) VALUES ($1, $2, $3, $4)",
        user_id, img_path, json.dumps(embeddings), len(embeddings)  # Convert to JSON string
        )
        
        await conn.close()
        return {
            "result": "registered",
            "user_id": user_id,
            "embeddings": len(embeddings),
            "faces_detected": len(faces_found)
        }
        
    except HTTPException:
        # Clean up on HTTP exceptions
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        if conn:
            await conn.close()
        raise
    except Exception as e:
        # Clean up on unexpected errors
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        if conn:
            await conn.close()
        logger.error(f"Unexpected error in registration: {e}")
        raise HTTPException(500, f"Registration failed: {str(e)}")


@app.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...)
):
    """Upload and analyze image for face detection"""
    conn = None
    img_path = None
    
    try:
        # Get client IP
        ip = getattr(request.client, "host", "unknown") if request.client else "unknown"
        
        # Save uploaded file
        img_path = save_upload(ACTOR_IMG_DIR, file.filename, file.file)
        
        
        # Read and validate image
        pil_img = Image.open(img_path).convert('RGB')
        img_np = np.array(pil_img)
        img= cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # cv2.imwrite(var,temp)
        # img = cv2.imread(var)
        if img is None:
            raise HTTPException(400, "Could not read image. Invalid or unsupported format.")
        
        # Detect faces
        try:
            detections = ensemble_detect_faces(img)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            if img_path and os.path.exists(img_path):
                os.remove(img_path)
            raise HTTPException(500, "Face detection failed")
            
        if not detections:
            os.remove(img_path)
            return {"result": "no_faces", "faces": 0, "matches": []}
        
        # Debug logging
        logger.info(f"Upload: Detected {len(detections)} faces")
        for i, det in enumerate(detections):
            emb = det.get("embedding", [])
            logger.info(f"Upload face {i}: embedding length = {len(emb)}")
        
        # Database operations
        conn = await get_db()
        
        # Handle bad actor record
        actor = await conn.fetchrow("SELECT actor_id FROM bad_actors WHERE ip_address=$1", ip)
        if not actor:
            actor_id = await conn.fetchval(
                "INSERT INTO bad_actors (ip_address, upload_count) VALUES ($1, 1) RETURNING actor_id", 
                ip
            )
        else:
            actor_id = actor["actor_id"]
            await conn.execute(
                "UPDATE bad_actors SET upload_count=upload_count+1, last_seen=NOW() WHERE actor_id=$1", 
                actor_id
            )
        
        # Record upload
        upload_id = await conn.fetchval(
            "INSERT INTO uploads (actor_id, file_path) VALUES ($1, $2) RETURNING upload_id", 
            actor_id, img_path
        )
        
        # Compare with registered faces
        profiles = await conn.fetch("SELECT user_id, embeddings FROM user_face_profiles")
        matches = []
        user_matches = {}
        
        logger.info(f"Comparing against {len(profiles)} registered profiles")
        
        for det in detections:
            det_emb = det.get("embedding", [])
            if not det_emb:
                continue
                
            det_emb_np = np.array(det_emb) if not isinstance(det_emb, np.ndarray) else det_emb
            
            for profile in profiles:
                embeddings_json = profile.get("embeddings", "[]")
                
                # Parse JSON string back to list
                try:
                    profile_embeddings = json.loads(embeddings_json) if isinstance(embeddings_json, str) else embeddings_json
                except (json.JSONDecodeError, TypeError):
                    logger.error(f"Failed to parse embeddings for user {profile['user_id']}")
                    continue

                if not profile_embeddings:
                    continue

                logger.info(f"User {profile['user_id']} has {len(profile_embeddings)} stored embeddings")
                
                # Find best similarity for this user
                best_sim = 0
                for ref_emb in profile_embeddings:
                    if not ref_emb:
                        continue

                    sim = cosine_sim(det_emb_np, ref_emb)
                    logger.info(f"User {profile['user_id']} similarity: {sim:.4f}")
                    
                    if sim > 0.6 and sim > best_sim:  # Lowered threshold
                        best_sim = sim
                        logger.info(f"New best match for user {profile['user_id']}: {best_sim:.4f}")

                # Track the best match for this user
                if best_sim > 0:
                    user_matches[profile["user_id"]] = max(user_matches.get(profile["user_id"], 0), best_sim)
                    logger.info(f"Final match for user {profile['user_id']}: {best_sim:.4f}")

        # Convert to list and record detections
        for user_id, sim in user_matches.items():
            matches.append((user_id, sim))
            try:
                await conn.execute(
                    "INSERT INTO detections (user_id, actor_id, upload_id, similarity) VALUES ($1, $2, $3, $4)",
                    user_id, actor_id, upload_id, sim
                )
                logger.info(f"✅ Recorded detection for user {user_id} with similarity {sim:.4f}")
            except Exception as e:
                logger.error(f"Failed to record detection: {e}")
        
        await conn.close()
        return {
            "result": "uploaded", 
            "faces": len(detections), 
            "matches": matches,
            "upload_id": upload_id
        }
        
    except HTTPException:
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        if conn:
            await conn.close()
        raise
    except Exception as e:
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        if conn:
            await conn.close()
        logger.error(f"Unexpected error in upload: {e}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/user/{user_id}/detections")
async def user_detections(user_id: int):
    """Get detection history for a user"""
    conn = None
    try:
        conn = await get_db()
        
        # Validate user exists
        user_exists = await conn.fetchval("SELECT 1 FROM users WHERE user_id=$1", user_id)
        if not user_exists:
            raise HTTPException(404, "User not found")
        
        # Get detections
        rows = await conn.fetch("""
        SELECT d.detection_id, d.similarity, d.detected_at, ba.ip_address, u.file_path
        FROM detections d
        JOIN bad_actors ba ON d.actor_id=ba.actor_id
        JOIN uploads u ON d.upload_id=u.upload_id
        WHERE d.user_id=$1 
        ORDER BY d.detected_at DESC
        """, user_id)
        
        await conn.close()
        
        result = []
        for r in rows:
            result.append({
                "detection_id": r["detection_id"],
                "similarity": float(r["similarity"]),
                "detected_at": r["detected_at"].isoformat() if r["detected_at"] else None,
                "uploader_ip": str(r["ip_address"]),
                "file": r["file_path"]
            })
        
        return {"user_id": user_id, "detections": result, "count": len(result)}
        
    except HTTPException:
        if conn:
            await conn.close()
        raise
    except Exception as e:
        if conn:
            await conn.close()
        logger.error(f"Error fetching user detections: {e}")
        raise HTTPException(500, f"Failed to fetch detections: {str(e)}")

@app.get("/image")
async def fetch_image(path: str):
    """Serve image files"""
    try:
        # Basic path validation to prevent directory traversal
        if ".." in path or not (path.startswith(USER_IMG_DIR) or path.startswith(ACTOR_IMG_DIR)):
            raise HTTPException(403, "Access denied")
        
        if os.path.exists(path) and os.path.isfile(path):
            return FileResponse(path)
        else:
            raise HTTPException(404, "File not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image {path}: {e}")
        raise HTTPException(500, "Failed to serve image")

@app.get("/")
async def root(request: Request):
    """Serve the main frontend page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test database connection
        conn = await get_db()
        await conn.execute("SELECT 1")
        await conn.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "running",
        "database": db_status,
        "models": {
            "insightface": face_app is not None,
            "yolo": yolo_model is not None
        },
        "directories": {
            "user_images": os.path.exists(USER_IMG_DIR),
            "actor_uploads": os.path.exists(ACTOR_IMG_DIR)
        }
    }