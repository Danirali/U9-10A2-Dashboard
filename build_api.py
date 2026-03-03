import pandas as pd
import numpy as np
from scipy import stats
import json

DATA_FILE = "footfall_data.csv"
print("Building Static APIs & Exports for GitHub Pages...")
df = pd.read_csv(DATA_FILE)

# 1. Build Dashboard & Zones Data
zone_metrics = {}
for zone, group in df.groupby("Zone"):
    zone_metrics[zone] = {
        "footfall": len(group),
        "dwell_min": round(group["DwellTime_s"].mean() / 60, 1),
        "engagement": round(group["EngagementScore"].mean(), 1),
        "conversion_rate": round(min(95, group["EngagementScore"].mean() * 1.1), 1) 
    }

dashboard_data = {
    "status": "success",
    "data": {
        "total_footfall": len(df),
        "avg_dwell_min": round(df["DwellTime_s"].mean() / 60, 1),
        "avg_engagement": round(df["EngagementScore"].mean(), 1),
        "ai_confidence": round(df["AI_Confidence"].mean() * 100, 1),
        "zones": zone_metrics
    }
}
with open('dashboard_data.json', 'w') as f:
    json.dump(dashboard_data, f)
print("✅ Created dashboard_data.json")

# 2. Build Unit 10 Statistics Data
dwell = df['DwellTime_s']
ent = df[df['Zone'] == 'Entrance']['DwellTime_s']
chk = df[df['Zone'] == 'Checkout']['DwellTime_s']
t_stat, p_val = stats.ttest_ind(ent, chk, equal_var=False) if len(ent)>1 and len(chk)>1 else (0,1)

lin_slope, lin_int, lin_r, _, _ = stats.linregress(df['DwellTime_s'], df['EngagementScore'])
valid_data = df[(df['DwellTime_s'] > 0) & (df['EngagementScore'] > 0)]
power_b, log_a, power_r, _, _ = stats.linregress(np.log(valid_data['DwellTime_s']), np.log(valid_data['EngagementScore']))

unit10_data = {
    "status": "success",
    "routine": {
        "mean": round(dwell.mean(), 1), "median": round(dwell.median(), 1), 
        "mode": round(dwell.mode()[0], 1), "std": round(dwell.std(), 1)
    },
    "non_routine": {
        "trimmed": round(stats.trim_mean(dwell, 0.1), 1), 
        "weighted": round(np.average(dwell, weights=df['AI_Confidence']), 1), 
        "iqr": round(dwell.quantile(0.75) - dwell.quantile(0.25), 1)
    },
    "ttest": {"t_stat": round(t_stat, 2), "p_val": round(p_val, 4)},
    "regression": {
        "linear": {"slope": round(lin_slope, 2), "int": round(lin_int, 2)},
        "power": {"a": round(np.exp(log_a), 2), "b": round(power_b, 2)}
    }
}
with open('unit10_data.json', 'w') as f:
    json.dump(unit10_data, f)
print("✅ Created unit10_data.json")

# 3. Pre-build XLSX for GitHub Pages Download
df.to_excel('Footfall_Analytics_Export.xlsx', index=False, sheet_name='Analytics')
print("✅ Created Footfall_Analytics_Export.xlsx")