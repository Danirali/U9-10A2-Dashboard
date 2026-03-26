import cv2
import numpy as np
import json
import os
from ultralytics import YOLO

# --- CONFIGURATION & FEASIBILITY ---
# Scale coordinates to match your dashboard image width (e.g., 1000px)
DASHBOARD_WIDTH = 1000 
JSON_PATH = 'static/heatmap_data.json'
INTENSITY = 70 

# Load YOLO model
model = YOLO("yolov8n.pt")

# Ensure static folder exists for the dashboard
if not os.path.exists('static'):
    os.makedirs('static')

# Initialize JSON file with an open bracket
with open(JSON_PATH, 'w') as f:
    f.write("[\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera")
    exit()

heatmap = None

# Decay setup: Fades old movement to keep data relevant
fps = cap.get(cv2.CAP_PROP_FPS) or 30
DECAY = 0.5 ** (1 / (fps * 60)) # 1 minute half-life

def save_to_json(x, y, intensity):
    """Saves optimized coordinate data for the web dashboard."""
    data = {
        "x": int(x), 
        "y": int(y), 
        "intensity": round(float(intensity), 2)
    }
    with open(JSON_PATH, 'a') as f:
        f.write(json.dumps(data) + ",\n")

def add_blob(heatmap_matrix, x, y, radius=50):
    """Updates matrix using OpenCV drawing (Fast) and saves data."""
    h, w = heatmap_matrix.shape
    
    # Use cv2.circle instead of nested loops for Distinction-level performance
    cv2.circle(heatmap_matrix, (x, y), radius, (INTENSITY), -1)
    
    # Get current intensity at the center point
    blob_intensity = heatmap_matrix[y, x]

    # SCALE COORDINATES: Map camera pixels to Dashboard pixels
    scale_factor = DASHBOARD_WIDTH / w
    scaled_x = x * scale_factor
    scaled_y = y * scale_factor

    save_to_json(scaled_x, scaled_y, blob_intensity)

print("AI Tracker Active. Press 'q' to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        if heatmap is None:
            heatmap = np.zeros((h, w), dtype=np.float32)

        # Run YOLO inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        if detections is not None:
            for box in detections:
                if int(box.cls[0]) == 0:  # Class 0 = Person
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    if 0 <= cx < w and 0 <= cy < h:
                        add_blob(heatmap, cx, cy)

        # Apply decay to handle temporal big data
        heatmap *= DECAY

        # Visualization for debugging
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Combine YOLO boxes and Heatmap
        annotated_frame = results[0].plot()
        final_output = cv2.addWeighted(annotated_frame, 0.7, heatmap_color, 0.3, 0)

        cv2.imshow("AI Tracker Feed", final_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Properly close the JSON array for the JavaScript fetch()
    with open(JSON_PATH, 'a') as f:
        f.write('{"x": 0, "y": 0, "intensity": 0}\n]')
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Data saved to {JSON_PATH}")