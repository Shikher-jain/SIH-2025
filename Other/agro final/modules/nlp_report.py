# modules/nlp_report.py
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load pretrained T5 model (can also fine-tune later on domain-specific data)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def generate_nlp_report(cell_id, pred_label, indices_dict, weather_dict):
    """
    Generate natural advisory report using a transformer model.
    Inputs:
        cell_id: string
        pred_label: int (0=Stressed,1=Moderate,2=Healthy)
        indices_dict: dict with NDVI, EVI, etc.
        weather_dict: dict with temperature, humidity, precipitation, etc.
    Returns:
        str: Natural language advisory
    """
    label_map = {0: "stressed", 1: "moderate", 2: "healthy"}
    label_text = label_map.get(pred_label, "unknown")

    # Create a textual prompt for the model
    prompt = f"Generate an agronomy advisory report for a field with crop condition {label_text}. "
    prompt += "Indices: "
    for k,v in indices_dict.items():
        if "_mean" in k:
            prompt += f"{k}:{v:.2f}, "
    prompt += "Weather: "
    for k,v in weather_dict.items():
        prompt += f"{k}:{v:.2f}, "

    # T5 expects a prefix task instruction
    input_text = "summarize: " + prompt

    # Tokenize
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        inputs,
        max_length=200,
        num_beams=4,
        early_stopping=True
    )
    report = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return report
