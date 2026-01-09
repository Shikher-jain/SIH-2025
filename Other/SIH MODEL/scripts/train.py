from model_cnn import build_model
from dataset import YieldHeatmapDataset

model = build_model()
train_gen = YieldHeatmapDataset("../data/data", "../data/mask", "../data/yield.csv")

model.fit(train_gen, epochs=25)
model.save("best_model.h5")
