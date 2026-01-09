import cv2
import os, glob
import numpy as np
from modules.indices import compute_indices_from_satellite, compute_indices_from_sensor
from modules.weather import parse_weather
from modules.model_utils import build_cnn_lstm_model
from modules.report_generator import generate_advisory_report
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns

# Load all cells
base_dir = "data"
cells = [os.path.join(base_dir,d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir,d))]

images = []
seqs = []
labels = []
cell_ids = []

for c in cells:
    cid = os.path.basename(c)
    sat_file = glob.glob(os.path.join(c,"*_satellite_*.tif"))[0]
    sensor_file = glob.glob(os.path.join(c,"*_sensor_*.tif"))[0]
    weather_file = glob.glob(os.path.join(c,"*_weather_*.json"))[0]

    # Indices
    sat_idx = compute_indices_from_satellite(sat_file)
    sensor_idx = compute_indices_from_sensor(sensor_file)
    indices = {**sat_idx, **sensor_idx}

    # Weather
    seq = parse_weather(weather_file)

    # Dummy image input for CNN: stack satellite+sensor bands
    with rasterio.open(sat_file) as src:
        img = src.read().astype(np.float32)
    with rasterio.open(sensor_file) as src:
        sensor_img = src.read().astype(np.float32)

    # sensor_img shape: (bands, H, W)
    resized_sensor = np.stack([cv2.resize(s, (img.shape[2], img.shape[1]), interpolation=cv2.INTER_LINEAR)
                            for s in sensor_img])
    img_stack = np.concatenate([img, resized_sensor], axis=0)

    # img_stack = np.concatenate([img,sensor_img],axis=0)
    # Resize to 64x64
    img_stack = np.transpose(img_stack,(1,2,0))
    img_stack = np.resize(img_stack,(64,64,img_stack.shape[2]))

    images.append(img_stack)
    seqs.append(seq)

    # Label based on NDVI mean (example)
    ndvi_mean = sat_idx['NDVI_mean']
    if ndvi_mean > 0.6: label=2
    elif ndvi_mean > 0.35: label=1
    else: label=0
    labels.append(label)
    cell_ids.append(cid)

images = np.array(images,dtype=np.float32)
seqs = np.array(seqs,dtype=np.float32)
labels = np.array(labels,dtype=np.int32)

# Train/Test split
from sklearn.model_selection import train_test_split
X_img_train,X_img_test,X_seq_train,X_seq_test,y_train,y_test,ids_train,ids_test = train_test_split(
    images,seqs,labels,cell_ids,test_size=0.2,random_state=42
)

# Build & train model
model = build_cnn_lstm_model(img_shape=images[0].shape, seq_len=24)
history = model.fit([X_img_train,X_seq_train],y_train,validation_data=([X_img_test,X_seq_test],y_test),epochs=15,batch_size=4)

# Predict test set
preds = np.argmax(model.predict([X_img_test,X_seq_test]),axis=1)

# Generate advisory reports
reports=[]
for i,cid in enumerate(ids_test):
    rep = generate_advisory_report(cid,preds[i],{**compute_indices_from_satellite(glob.glob(os.path.join(base_dir,cid,"*_satellite_*.tif"))[0]),**compute_indices_from_sensor(glob.glob(os.path.join(base_dir,cid,"*_sensor_*.tif"))[0])},{'temperature': X_seq_test[i][:,0].mean(),'humidity': X_seq_test[i][:,1].mean(),'precipitation': X_seq_test[i][:,2].mean(),'windSpeed': X_seq_test[i][:,3].mean(),'pressure': X_seq_test[i][:,4].mean()})

    reports.append(rep)
    print("-----\n",rep)

# Generate heatmap (PNG)
plt.figure(figsize=(12,6))
sns.heatmap(np.array([preds]).reshape(1,-1),cmap='RdYlGn',cbar=True)
plt.title("Predicted Crop Condition Heatmap (0=Stressed,1=Moderate,2=Healthy)")
plt.savefig("outputs/health_map.png")
plt.show()
