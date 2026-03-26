import cv2
import numpy as np
import pandas as pd
import time
import os
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Dashboard Mapping
MAP_W, MAP_H = 1000, 800 # Match your storelayout.jpg dimensions

# Tracker for Dwell Time
track_history = {} 
final_logs = [] # We only move data here when a person LEAVES

def get_zone(x, y, w, h):
    if x < w/3: return "Entrance"
    elif x > 2*w/3: return "Checkout"
    else: return "Aisle_1"

print("AI Tracker Running... Press 'q' to save and exit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape

    # Use 'persist=True' for tracking across frames
    results = model.track(frame, persist=True, verbose=False) 
    
    current_ids = []
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confidences):
            current_ids.append(track_id)
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            
            # SCALE coordinates to match the dashboard image
            scaled_x = int(cx * (MAP_W / w))
            scaled_y = int(cy * (MAP_H / h))

            if track_id not in track_history:
                track_history[track_id] = {
                    'start': time.time(), 
                    'hits': 1, 
                    'last_seen': time.time(),
                    'zone': get_zone(cx, cy, w, h),
                    'coords': (scaled_x, scaled_y),
                    'conf': float(conf)
                }
            else:
                track_history[track_id]['hits'] += 1
                track_history[track_id]['last_seen'] = time.time()

    # --- BIG DATA CLEANING LOGIC ---
    # Check for IDs that have left the frame (idle for > 2 seconds)
    ids_to_remove = []
    for tid, info in track_history.items():
        if time.time() - info['last_seen'] > 2.0:
            # Person has left! Calculate final stats for this record
            dwell = time.time() - info['start']
            # Engagement formula: (Dwell weight 0.7) + (Movement weight 0.3)
            engagement = (dwell * 0.7) + (info['hits'] * 0.01)

            final_logs.append({
                "DetectionID": f"ANON-{tid}",
                "Timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Zone": info['zone'],
                "DwellTime_s": round(dwell, 2),
                "EngagementScore": round(min(engagement, 100), 2), # Cap at 100
                "AI_Confidence": round(info['conf'], 2),
                "X": info['coords'][0],
                "Y": info['coords'][1]
            })
            ids_to_remove.append(tid)

    for tid in ids_to_remove:
        del track_history[tid]

    # Visual Feedback
    cv2.imshow("AI Vision Capture - GDPR MODE", results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# --- PROJECT CLOSURE (TASK 5/6) ---
# Save the structured data
if final_logs:
    df = pd.DataFrame(final_logs)
    df.to_csv("AI Vision/footfall_data.csv", index=False)
    print(f"Successfully captured {len(df)} unique customer interactions.")
else:
    print("No data captured.")

cap.release()
cv2.destroyAllWindows()