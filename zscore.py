import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats

# Step 1: Generate Data

np.random.seed(42)

# Normal data
normal_data = np.random.normal(loc=50, scale=5, size=100)

# Add outliers
outliers = np.array([100, 110, 120])

# Combine
data = np.concatenate((normal_data, outliers))

# Convert to 2D for Isolation Forest
data_2d = data.reshape(-1, 1)

# Step 2: Z-score Method

z_scores = stats.zscore(data)

threshold = 3
z_outliers = data[np.abs(z_scores) > threshold]

print("Z-score Outliers:", z_outliers)

# Step 3: Isolation Forest

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data_2d)

pred = model.predict(data_2d)

# -1 = anomaly
if_outliers = data[pred == -1]

print("Isolation Forest Outliers:", if_outliers)

# Step 4: Plot Results

plt.figure()

# All data
plt.scatter(range(len(data)), data, label="Data")

# Z-score outliers
plt.scatter(np.where(np.isin(data, z_outliers)),
            z_outliers, color='red', label="Z-score Outliers")

# Isolation Forest outliers
plt.scatter(np.where(np.isin(data, if_outliers)),
            if_outliers, color='green', label="IF Outliers")

plt.legend()
plt.title("Outlier Detection (Z-score vs Isolation Forest)")
plt.xlabel("Index")
plt.ylabel("Value")

plt.show()