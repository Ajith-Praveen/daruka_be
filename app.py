"""
DARUKA-VLAB BACKEND
YOLOv8 Fire Detection API
Flask server for real-time fire detection
"""

import os
import base64
import io
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# --------------------------------------------------
# Flask App Setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# YOLOv8 Model Setup
# --------------------------------------------------
MODEL_PATH = "fire-yolov8.pt"   # keep model in same folder
DEFAULT_THRESHOLD = 0.5

try:
    model = YOLO(MODEL_PATH)
    print("[BACKEND] Custom fire model loaded")
except Exception as e:
    print("[BACKEND] Custom model not found, loading yolov8n.pt")
    print(e)
    model = YOLO("yolov8n.pt")

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": "Daruka-VLab Fire Detection API",
        "version": "1.0.0"
    })


@app.route("/api/detect", methods=["POST"])
def detect_fire():
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "Image not provided"}), 400

        # Threshold validation
        threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
        threshold = max(0.1, min(threshold, 0.9))

        # Decode base64 image
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize image (IMPORTANT for low-RAM servers)
        image = image.resize((640, 640))

        # PIL → OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # YOLO Inference
        results = model(image_cv, conf=threshold)

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                label = model.names[class_id]

                detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

        return jsonify({
            "detections": detections,
            "count": len(detections),
            "timestamp": datetime.utcnow().isoformat()
        })

    except Exception as e:
        print("[BACKEND] Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/camera-test", methods=["GET"])
def camera_test():
    return jsonify({
        "status": "ok",
        "message": "Backend ready for fire detection",
        "model_loaded": model is not None
    })


# --------------------------------------------------
# Entry Point (Cloud Compatible)
# --------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # 7860 = Hugging Face default
    print(f"[BACKEND] Server starting on port {port}")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
