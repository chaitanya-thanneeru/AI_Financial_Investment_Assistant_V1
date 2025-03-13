import torch
from transformers import AutoModelForSequenceClassification

# Load Model
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Convert to INT8 Quantized Model
quantized_model = model.to(torch.float16)
quantized_model.save_pretrained("models/finbert_quantized")

print("✅ Model quantized and saved!")
