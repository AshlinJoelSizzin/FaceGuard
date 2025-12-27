import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import argparse

def load_models(yolo_weights_path=r"C:\Users\surya\Downloads\yolov8x-face-lindevs.pt"):
    """Load face detection models"""
    print("Loading InsightFace model...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    print("Loading YOLOv8-face model...")
    if os.path.exists(yolo_weights_path):
        yolo_model = YOLO(yolo_weights_path)
    else:
        print(f"Warning: YOLO weights file not found: {yolo_weights_path}")
        yolo_model = None
    
    return face_app, yolo_model

def detect_faces_ensemble(image, face_app, yolo_model=None):
    """Detect faces using ensemble of SCRFD and YOLOv8"""
    if image is None or image.size == 0:
        return []
    
    detections = []
    
    # SCRFD detection
    faces_scrfd = face_app.get(image)
    
    # YOLOv8 detection (if available)
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
                            if x2 > x1 and y2 > y1:
                                boxes_yolo.append((x1, y1, x2, y2))
        except Exception as e:
            print(f"YOLO detection failed: {e}")
    
    # Process SCRFD detections
    for face in faces_scrfd:
        try:
            bbox = face.get("bbox", [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
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
            print(f"Error processing SCRFD face: {e}")
            continue
    
    return detections

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    img_copy = image.copy()
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add score if available
        score = det.get("score", 0)
        if score > 0:
            cv2.putText(img_copy, f"{score:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return img_copy

def main():
    parser = argparse.ArgumentParser(description="Face detection script")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="output.jpg", help="Path to output image")
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"Error: Input image not found at {args.image}")
        return
    
    # Load models
    face_app, yolo_model = load_models()
    
    # Read image
    image = cv2.imread(args.image)
    if image is None:
        print("Error: Could not read image")
        return
    
    print("Detecting faces...")
    detections = detect_faces_ensemble(image, face_app, yolo_model)
    
    print(f"Found {len(detections)} faces")
    
    # Draw detections
    result_img = draw_detections(image, detections)
    
    # Save result
    cv2.imwrite(args.output, result_img)
    print(f"Result saved to {args.output}")
    
    # Show result (optional)
    cv2.imshow("Face Detection Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()