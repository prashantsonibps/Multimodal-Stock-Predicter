import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load and combine datasets
def load_data(tickers, data_dir="data/final"):
    dfs = []
    for ticker in tickers:
        path = os.path.join(data_dir, f"{ticker}_final.csv")
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Define MLP model
class FusionMLP(nn.Module):
    def __init__(self, input_dim):
        super(FusionMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Main training pipeline
def main():
    tickers = ['TSLA', 'MSFT', 'AAPL']
    df = load_data(tickers)


    print("Feature columns:", X.columns.tolist())
    print("Input dim:", X.shape[1])
    

    # Drop non-feature columns
    X = df.drop(columns=['Date', 'Close', 'next_close', 'target'])
    y = df['target']

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")


    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Convert to PyTorch tensors
    def to_tensor(x, y): return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(to_tensor(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(to_tensor(X_val, y_val), batch_size=32)
    test_loader = DataLoader(to_tensor(X_test, y_test), batch_size=32)

    # Initialize model
    model = FusionMLP(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_val_loss = float('inf')
    patience, patience_counter = 5, 0

    for epoch in range(50):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb)
                val_loss += criterion(preds, yb).item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/fusion_mlp.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Evaluation
    model.load_state_dict(torch.load("models/fusion_mlp.pt"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = (model(xb) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    print("\n[📊] Classification Report:")
    print(classification_report(all_labels, all_preds))

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()

