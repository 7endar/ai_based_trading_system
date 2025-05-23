import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt  # Görselleştirme için eklendi

# --- VERİ YÜKLEME & ÖN İŞLEME ---
conn = sqlite3.connect("btc_dataset.db")
price_df = pd.read_sql_query("SELECT * FROM ohlcv_5m", conn)
indicators_df = pd.read_sql_query("SELECT * FROM indicators_5m", conn)

df = pd.merge(indicators_df, price_df, on='timestamp', how='inner')
df = df.sort_values("timestamp").reset_index(drop=True)

features = ['rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'obv', 'ema',
            'open', 'high', 'low', 'close', 'volume']

df.dropna(inplace=True)

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

SEQ_LEN = 288  # geçmiş veri uzunluğu

# Yeni zaman aralıkları eklendi
PREDICTION_HORIZONS = {
    '5m': 1,    # 1 adım sonrası (5 dakika)
    '15m': 3,   # 3 adım sonrası (15 dakika)
    '30m': 6,   # 6 adım
    '1h': 12,   # 12 adım (60 dakika)
    '4h': 48,   # 48 adım (4 saat)
    '24h': 288  # 288 adım (24 saat)
}


# --- DATASET TANIMI (GÜNCELLENDİ) ---
# --- DATASET TANIMI (DÜZELTİLMİŞ) ---
class CryptoDataset(Dataset):
    def __init__(self, df, features):
        self.data = df[features].values.astype(np.float32)
        self.labels = df['close'].values.astype(np.float32)

    def __len__(self):
        # SEQ_LEN + en uzun horizon için yeterli veri kaldığından emin ol
        return max(0, len(self.data) - SEQ_LEN - max(PREDICTION_HORIZONS.values()))

    def __getitem__(self, idx):
        x = self.data[idx:idx + SEQ_LEN]
        y = np.array([
            self.labels[idx + SEQ_LEN + 1 - 1],   # 5m sonrası (1 adım)
            self.labels[idx + SEQ_LEN + 3 - 1],   # 15m sonrası (3 adım)
            self.labels[idx + SEQ_LEN + 6 - 1],   # 30m sonrası (6 adım)
            self.labels[idx + SEQ_LEN + 12 - 1],  # 1h sonrası (12 adım)
            self.labels[idx + SEQ_LEN + 48 - 1],  # 4h sonrası (48 adım)
            self.labels[idx + SEQ_LEN + 288 - 1]  # 24h sonrası (288 adım)
        ], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y)


# --- MODEL TANIMI (DEĞİŞMEDİ) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


class TransformerRegressor(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.regressor = nn.Linear(d_model, len(PREDICTION_HORIZONS))  # Tüm horizonlar için

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x[-1]
        return self.regressor(x)


# --- VERİ BÖLME (VALIDATION EKLENDİ) ---
train_val_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, shuffle=False)  # %60 train, %20 val, %20 test

train_dataset = CryptoDataset(train_df, features)
val_dataset = CryptoDataset(val_df, features)
test_dataset = CryptoDataset(test_df, features)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- EĞİTİM (VALIDATION LOSS EKLENDİ) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerRegressor(num_features=len(features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    epoch_train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss / len(train_loader))

    # Validation
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            epoch_val_loss += loss.item()
    val_losses.append(epoch_val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_losses[-1]:.6f} - Val Loss: {val_losses[-1]:.6f}")

# Loss grafiği (Overfitting kontrolü)
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# --- TEST VE PERFORMANS DEĞERLENDİRME (TÜM HORIZONLAR İÇİN) ---
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

# Her horizon için metrikler
for i, label in enumerate(PREDICTION_HORIZONS.keys()):
    mse = mean_squared_error(all_targets[:, i], all_preds[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
    r2 = r2_score(all_targets[:, i], all_preds[:, i])
    print(f"\nPerformance on {label} prediction:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R2: {r2:.6f}")


# --- TAHMİN FONKSİYONU (GÜNCELLENDİ) ---
def multi_horizon_predict(model, df, features, scaler, seq_len):
    model.eval()
    window = df[features].values[-seq_len:].astype(np.float32)
    x = torch.tensor(window).unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred_norm = model(x).cpu().numpy().flatten()

    dummy = np.zeros((1, len(features)))
    preds = {}

    # Mevcut fiyat
    latest_close_scaled = df['close'].values[-1]
    dummy[0, features.index("close")] = latest_close_scaled
    preds["current"] = scaler.inverse_transform(dummy)[0, features.index("close")]

    # Tahminler
    for i, (label, norm_val) in enumerate(zip(PREDICTION_HORIZONS.keys(), y_pred_norm)):
        dummy[0, features.index("close")] = norm_val
        preds[label] = scaler.inverse_transform(dummy)[0, features.index("close")]

    return preds


# Örnek tahmin
preds = multi_horizon_predict(model, df, features, scaler, SEQ_LEN)
print("\nSon Fiyat ve Tahminler:")
for horizon, price in preds.items():
    print(f"{horizon:>5}: ${price:.4f}")