import numpy as np
import tensorflow as tf
import joblib
import time
import serial  # For IoT sensor data (if applicable)

# Load trained model and scaler
model = tf.keras.models.load_model("vibration_model.h5")
scaler = joblib.load("scaler.pkl")

# Simulated real-time sensor data stream
def get_sensor_data():
    return np.random.normal(0, 0.5, 50)  # Replace with actual sensor input

while True:
    # Read real-time vibration data
    new_data = get_sensor_data()
    new_data_scaled = scaler.transform(new_data.reshape(-1, 1)).reshape(1, 50, 1)

    # Predict anomaly score
    prediction = model.predict(new_data_scaled)[0][0]

    # Alert if vibration exceeds threshold
    if prediction > 0.7:  # Adjust threshold as needed
        print("⚠️ Warning: Potential machine fault detected!")

    # Wait before next inference (simulate real-time processing)
    time.sleep(1)
