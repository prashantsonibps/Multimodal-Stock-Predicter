import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np

from train_model import FusionMLP

# Load model
model = FusionMLP(input_dim=8)  # Adjust if your feature count changes
model.load_state_dict(torch.load("models/fusion_mlp.pt"))
model.eval()

# Load scaler
scaler = joblib.load("models/scaler.pkl")

# Define prediction function
def predict(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred = model(input_tensor)
        label = "UP" if pred.item() > 0.5 else "DOWN"
        print(f"[📈] Predicted: {label} (score={pred.item():.4f})")
        return label

# Example usage
if __name__ == "__main__":
    example_input = {
        "SMA_20": 22.13,
        "SMA_50": 21.91,
        "RSI": 68.5,
        "MACD": 0.07,
        "MACD_Signal": -0.02,
        "Volume": 36723500,
        "avg_sentiment": 0.095008,
        "Open": 22.16  # if included
    }
    predict(example_input)
