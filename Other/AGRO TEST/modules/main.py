import numpy as np
from modules.data_loader import load_all_cells
from modules.feature_engine import compute_indices, aggregate_indices, resize_image, assign_label
from modules.model_utils import build_cnn_lstm_model
from modules.predictor import predict_crop
from modules.report_generator import generate_advisory_report
from modules.dashboard import generate_cell_dashboard
from sklearn.model_selection import train_test_split

# 1️⃣ Load data
data, cell_ids = load_all_cells("HectFarm-1")

images, sequences, labels, agg_indices_list = [],[],[],[]
for d in data:
    sat_img = resize_image(d['sat'])
    images.append(sat_img)
    seq_arr = np.zeros((24,5),dtype=np.float32)
    for i in range(min(24,len(d['weather'].get('seq24',[])))):
        w = d['weather']['seq24'][i]
        seq_arr[i,0]=w.get('temperature',28)
        seq_arr[i,1]=w.get('humidity',50)
        seq_arr[i,2]=w.get('precipitation',0)
        seq_arr[i,3]=w.get('windSpeed',2)
        seq_arr[i,4]=w.get('pressure',1005)
    sequences.append(seq_arr)
    inds = aggregate_indices(compute_indices(d['sat']))
    agg_indices_list.append(inds)
    labels.append(assign_label(inds))

images = np.stack(images)
sequences = np.stack(sequences)
labels = np.array(labels)

# 2️⃣ Train/Test split
X_img_train,X_img_test,X_seq_train,X_seq_test,y_train,y_test,ids_train,ids_test=train_test_split(
    images,sequences,labels,cell_ids,test_size=0.2,random_state=42
)

# 3️⃣ Train model
model = build_cnn_lstm_model(img_shape=images[0].shape, seq_len=24)
history = model.fit([X_img_train,X_seq_train],y_train,
                    validation_data=([X_img_test,X_seq_test],y_test),
                    epochs=15,batch_size=4)

# 4️⃣ Predict
pred_json = predict_crop(model,X_img_test,X_seq_test,ids_test)

# 5️⃣ Generate advisory reports & dashboards
for i,d in enumerate(pred_json):
    cid = d['cell_id']
    report = generate_advisory_report(cid,d['pred_label'],agg_indices_list[i],{'temperature':X_seq_test[i,:,0].mean(),
                                                                             'humidity':X_seq_test[i,:,1].mean(),
                                                                             'precipitation':X_seq_test[i,:,2].mean(),
                                                                             'windSpeed':X_seq_test[i,:,3].mean(),
                                                                             'pressure':X_seq_test[i,:,4].mean()})
    d['report']=report
    generate_cell_dashboard(cid,d['pred_label'],agg_indices_list[i],{},report)

# 6️⃣ Print sample
for r in pred_json[:3]:
    print(r)
