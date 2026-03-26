import pandas as pd
import json
import matplotlib.pyplot as plt

file = open('heatmap.txt','r')
dict = json.loads(file.read())

df = pd.DataFrame(dict)
df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')

plt.figure(figsize=(9, 7))
scatter = plt.scatter(
    df['x'], df['y'],
    c=df['intensity'],          
    cmap='YlOrRd',              
    s=df['intensity'],                        
    edgecolor='k', alpha=0.7
)

plt.colorbar(scatter, label='Intensity')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot – colour = intensity')
plt.gca().invert_yaxis()   # optional: match image‑coordinate systems
plt.tight_layout()
plt.show()
