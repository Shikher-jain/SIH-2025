# main.py
from modules.dataset_builder import build_dataset
from modules.train_model import train_model
from generate_advisory import generate_advisory
import os

DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("✅ Step 1: Building dataset...")
csv_file = build_dataset(DATA_DIR)

print("✅ Step 2: Training model...")
model_file = train_model(epochs=15, batch_size=8)

print("✅ Step 3: Generating advisory reports and heatmap...")
generate_advisory()

print("✅ All steps completed. Check outputs/ for CSV, model, advisories and heatmap.")
