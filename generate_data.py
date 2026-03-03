import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta

NUM_RECORDS = 5000
FILENAME = "footfall_data.csv"
start_date = datetime.now() - timedelta(days=7)
data = []

print(f"Generating {NUM_RECORDS} rows of synthetic BTEC footfall data...")

for i in range(NUM_RECORDS):
    det_id = f"ANON-{uuid.uuid4().hex[:6]}"
    ts = start_date + timedelta(seconds=np.random.randint(0, 7 * 24 * 60 * 60))
    zone = np.random.choice(["Entrance", "Checkout", "High-Conv"], p=[0.25, 0.35, 0.40])
    
    is_staff = np.random.random() < 0.02 # 2% staff (Outliers)
    if is_staff:
        dwell = np.random.uniform(3600, 14400) 
    else:
        if zone == "Entrance": dwell = np.random.normal(30, 10)
        elif zone == "Checkout": dwell = np.random.normal(120, 45)
        else: dwell = np.random.normal(300, 120)
            
    dwell = max(5, round(dwell, 2))
    
    if is_staff: engage = np.random.uniform(20, 40)
    else: engage = min(100, max(0, round(15 * (dwell ** 0.3) + np.random.normal(0, 5), 2)))
        
    conf = np.random.uniform(0.65, 0.88) if zone == "Entrance" else np.random.uniform(0.85, 0.99)
        
    data.append({
        "DetectionID": det_id, "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "Zone": zone, "DwellTime_s": dwell, "EngagementScore": engage, "AI_Confidence": round(conf, 2)
    })

df = pd.DataFrame(data).sort_values(by="Timestamp")
df.to_csv(FILENAME, index=False)
print("✅ Data generation complete!")