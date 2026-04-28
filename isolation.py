# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Step 1: Generate sample data
rng = np.random.RandomState(42)

# Normal data (cluster)
X_normal = 0.3 * rng.randn(100, 2)

# Outliers (far away points)
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# Combine data
X = np.r_[X_normal, X_outliers]

# Step 2: Train Isolation Forest model
model = IsolationForest(contamination=0.15, random_state=42)
model.fit(X)

# Step 3: Predict anomalies
y_pred = model.predict(X)

# -1 = anomaly, 1 = normal
print("Predictions:\n", y_pred)

# Step 4: Separate normal and anomalies
normal_points = X[y_pred == 1]
anomalies = X[y_pred == -1]

# Step 5: Plot results
plt.scatter(normal_points[:, 0], normal_points[:, 1], c='blue', label='Normal')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', label='Anomaly')
plt.legend()
plt.title("Isolation Forest Anomaly Detection")
plt.show()