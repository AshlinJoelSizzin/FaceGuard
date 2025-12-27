import os
import shutil
import numpy as np
import cv2
import asyncpg
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import albumentations as A
import hashlib
import bcrypt

DB_URL = "postgresql://postgres:Password@localhost:5432/face_detection_db"  # <-- PUT your password!
USER_IMG_DIR = "user_images"
ACTOR_IMG_DIR = "actor_uploads"
YOLOV8_FACE_WEIGHTS = "yolov8x-face-lindevs.pt"
os.makedirs(USER_IMG_DIR, exist_ok=True)
os.makedirs(ACTOR_IMG_DIR, exist_ok=True)

# --- ML MODELS ---
print("Loading InsightFace models...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

print("Loading YOLOv8-face ONNX...")
yolo_model = YOLO(YOLOV8_FACE_WEIGHTS)

AUGMENTER = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(p=0.2),
    A.RandomGamma(p=0.4),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=12, p=0.4)
])

async def get_db():
    return await asyncpg.connect(DB_URL)

def hash_img(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def save_upload(upload_dir, filename, file):
    os.makedirs(upload_dir, exist_ok=True)
    dest = os.path.join(upload_dir, filename)
    with open(dest, "wb") as buf:
        shutil.copyfileobj(file, buf)
    return dest

def iou_bbox(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(a_area + b_area - inter_area + 1e-6)

def ensemble_detect_faces(image: np.ndarray):
    if image is None:
        return []
    # SCRFD
    faces_scrfd = face_app.get(image)
    # YOLOv8
    yolo_results = yolo_model.predict(source=image, conf=0.5, verbose=False)
    boxes_yolo = []
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.flatten())
            boxes_yolo.append((x1, y1, x2, y2))
    detections = []
    for face in faces_scrfd:
        x1, y1, x2, y2 = map(int, face["bbox"])
        overlap = any(iou_bbox((x1,y1,x2,y2), yb) > 0.4 for yb in boxes_yolo)
        if overlap or not boxes_yolo:
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "aligned": face["aligned"],
                "embedding": face["embedding"],
                "score": float(face["det_score"])
            })
    # Optionally add YOLO boxes only if not overlapping any SCRFD
    for bx in boxes_yolo:
        if not any(iou_bbox(bx, tuple(map(int, f["bbox"]))) > 0.4 for f in faces_scrfd):
            crop = image[bx[1]:bx[3], bx[0]:bx[2]]
            if crop is None or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue
            crop = cv2.resize(crop, (112,112))
            embedding = face_app.model.get_feat(crop)
            detections.append({
                "bbox": bx,
                "aligned": crop,
                "embedding": embedding.tolist(),
                "score": 0.5
            })
    return detections

def augment_face(face_img: np.ndarray, n=4):
    return [AUGMENTER(image=face_img)["image"] for _ in range(n)]

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

app = FastAPI()

@app.post("/register")
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        img_path = save_upload(USER_IMG_DIR, f"{username}_{image.filename}", image.file)
        img = cv2.imread(img_path)
        if img is None:
            os.remove(img_path)
            raise HTTPException(400, "Could not read image. Invalid or unsupported format.")
        faces_found = ensemble_detect_faces(img)
        if not faces_found:
            os.remove(img_path)
            raise HTTPException(400, "No face detected.")
        embeddings = []
        for face in faces_found or []:
            faces_to_embed = [face["aligned"]]
            faces_to_embed += augment_face(face["aligned"])
            for fimg in faces_to_embed:
                emb = getattr(face_app, 'model').get_feat(fimg)
                embeddings.append(emb.tolist())
        conn = await get_db()
        hashed_password = hash_password(password)
        await conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3)",
            username, email, hashed_password.decode('utf-8')
        )
        user_id = await conn.fetchval("SELECT user_id FROM users WHERE username=$1", username)
        await conn.execute(
            "INSERT INTO user_face_profiles (user_id, image_path, embeddings, augment_count) VALUES ($1, $2, $3, $4)",
            user_id, img_path, embeddings, len(embeddings)
        )
        await conn.close()
        return {"result": "registered", "user_id": user_id, "embeddings": len(embeddings)}
    except Exception as e:
        raise HTTPException(500, f"Registration failed: {str(e)}")

@app.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        ip = getattr(request.client, "host", "0.0.0.0")
        img_path = save_upload(ACTOR_IMG_DIR, file.filename, file.file)
        img = cv2.imread(img_path)
        if img is None:
            os.remove(img_path)
            raise HTTPException(400, "Could not read image. Invalid or unsupported format.")
        detections = ensemble_detect_faces(img)
        if not detections:
            os.remove(img_path)
            return {"result": "no_faces"}
        conn = await get_db()
        actor = await conn.fetchrow("SELECT actor_id FROM bad_actors WHERE ip_address=$1", ip)
        if not actor:
            actor_id = await conn.fetchval(
                "INSERT INTO bad_actors (ip_address, upload_count) VALUES ($1, 1) RETURNING actor_id", ip)
        else:
            actor_id = actor["actor_id"]
            await conn.execute(
                "UPDATE bad_actors SET upload_count=upload_count+1, last_seen=NOW() WHERE actor_id=$1", actor_id)
        upload_id = await conn.fetchval(
            "INSERT INTO uploads (actor_id, file_path) VALUES ($1, $2) RETURNING upload_id", actor_id, img_path)
        profiles = await conn.fetch("SELECT user_id, embeddings FROM user_face_profiles")
        matches = []
        for det in detections:
            det_emb = det["embedding"] if isinstance(det["embedding"], np.ndarray) else np.array(det["embedding"])
            for profile in profiles:
                for ref_emb in profile["embeddings"]:
                    sim = cosine_sim(det_emb, np.array(ref_emb))
                    if sim > 0.85:
                        matches.append((profile["user_id"], sim))
                        await conn.execute(
                            "INSERT INTO detections (user_id, actor_id, upload_id, similarity) VALUES ($1, $2, $3, $4)",
                            profile["user_id"], actor_id, upload_id, sim)
        await conn.close()
        return {"result": "uploaded", "faces": len(detections), "matches": matches}
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/user/{user_id}/detections")
async def user_detections(user_id: int):
    try:
        conn = await get_db()
        rows = await conn.fetch("""
        SELECT d.detection_id, d.similarity, d.detected_at, ba.ip_address, u.file_path
        FROM detections d
        JOIN bad_actors ba ON d.actor_id=ba.actor_id
        JOIN uploads u ON d.upload_id=u.upload_id
        WHERE d.user_id=$1 ORDER BY d.detected_at DESC
        """, user_id)
        await conn.close()
        result = []
        for r in rows:
            result.append({
                "detection_id": r["detection_id"],
                "similarity": r["similarity"],
                "detected_at": r["detected_at"],
                "uploader_ip": str(r["ip_address"]),
                "file": r["file_path"]
            })
        return result
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch detections: {str(e)}")

@app.get("/image")
async def fetch_image(path: str):
    if os.path.exists(path): 
        return FileResponse(path)
    else: 
        raise HTTPException(404, "File not found.")

@app.get("/")
async def root():
    return {"status": "running"}
