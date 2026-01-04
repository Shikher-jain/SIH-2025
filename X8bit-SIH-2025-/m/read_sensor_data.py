# read_sensor_data.py

import pandas as pd

# Load sensor data CSV
sensor_data = pd.read_csv('sensor_data.csv')

print("Sensor Data Sample:")
print(sensor_data.head())

# Example: plot soil moisture
import matplotlib.pyplot as plt

plt.plot(sensor_data['Date'], sensor_data['SoilMoisture'])
plt.xticks(rotation=45)
plt.ylabel("Soil Moisture (%)")
plt.title("Soil Moisture Over Time")
plt.tight_layout()
plt.show()
