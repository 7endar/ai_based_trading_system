import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

# --- VERİ YÜKLEME & ÖN İŞLEME ---

conn = sqlite3.connect("btc_dataset.db")  # Kendi dosya adını kullan

price_df = pd.read_sql_query("SELECT * FROM ohlcv_1h", conn)
indicators_df = pd.read_sql_query("SELECT * FROM indicators_1h", conn)

df = pd.merge(indicators_df, price_df, on='timestamp', how='inner')
df = df.sort_values("timestamp").reset_index(drop=True)

features = ['rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'obv', 'ema',
            'open', 'high', 'low', 'close', 'volume']

df.dropna(inplace=True)

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

SEQ_LEN = 500  # geçmiş veri uzunluğu
PRED_OFFSET = 24  # 24 saat sonrası hedef


# --- DATASET TANIMI ---

class CryptoDataset(Dataset):
    def __init__(self, df, features):
        self.data = df[features].values.astype(np.float32)
        self.labels = df['close'].values.astype(np.float32)

    def __len__(self):
        return max(0, len(self.data) - SEQ_LEN - PRED_OFFSET)

    def __getitem__(self, idx):
        x = self.data[idx:idx + SEQ_LEN]
        y_1h = self.labels[idx + SEQ_LEN + 1 - 1]     # 1 saat sonrası
        y_4h = self.labels[idx + SEQ_LEN + 4 - 1]     # 4 saat sonrası
        y_24h = self.labels[idx + SEQ_LEN + 24 - 1]   # 24 saat sonrası
        y = np.array([y_1h, y_4h, y_24h], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y)


# --- MODEL TANIMI ---

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
        self.regressor = nn.Linear(d_model, 3)  # 3 hedef: 1h, 4h, 24h

    def forward(self, x):  # x: (B, T, F)
        x = self.input_proj(x)  # -> (B, T, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Transformer expects (T, B, E)
        x = self.transformer(x)
        x = x[-1]  # last timestep output
        return self.regressor(x)  # -> (B, 3)


# --- TRAIN / TEST VERİLERİ BÖLME ---

train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

train_dataset = CryptoDataset(train_df, features)
test_dataset = CryptoDataset(test_df, features)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --- CİHAZ & MODEL HAZIRLIK ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerRegressor(num_features=len(features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# --- EĞİTİM DÖNGÜSÜ ---

EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss/len(train_loader):.6f}")


# --- TEST VE PERFORMANS DEĞERLENDİRME ---

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

# Metrik hesaplama (her hedef için)
for i, label in enumerate(["1h", "4h", "24h"]):
    mse = mean_squared_error(all_targets[:, i], all_preds[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets[:, i], all_preds[:, i])
    r2 = r2_score(all_targets[:, i], all_preds[:, i])
    print(f"\nPerformance on {label} prediction:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R2: {r2:.6f}")


# --- MODEL KAYDETME ---

torch.save(model.state_dict(), "transformer_regressor.pth")
print("\nModel 'transformer_regressor.pth' olarak kaydedildi.")


# --- TEK SEFERLİK TAHMİN FONKSİYONU ---

def multi_horizon_predict(model, df, features, scaler, seq_len):
    model.eval()
    window = df[features].values[-seq_len:].astype(np.float32)
    x = torch.tensor(window).unsqueeze(0).to(device)  # (1, seq_len, features)

    with torch.no_grad():
        y_pred_norm = model(x).cpu().numpy().flatten()

    dummy = np.zeros((1, len(features)))
    preds = {}
    for i, (label, norm_val) in enumerate(zip(["1h", "4h", "24h"], y_pred_norm)):
        dummy[0, features.index("close")] = norm_val
        preds[label] = scaler.inverse_transform(dummy)[0, features.index("close")]

    latest_close_scaled = df['close'].values[-1]
    dummy[0, features.index("close")] = latest_close_scaled
    preds["current"] = scaler.inverse_transform(dummy)[0, features.index("close")]
    return preds


# --- KAYDEDİLEN MODEL İLE ÖRNEK TAHMİN ---

# Yeni model nesnesi oluştur ve ağırlıkları yükle
loaded_model = TransformerRegressor(num_features=len(features)).to(device)
loaded_model.load_state_dict(torch.load("transformer_regressor.pth", map_location=device))

preds = multi_horizon_predict(loaded_model, df, features, scaler, SEQ_LEN)

print(f"\nSon fiyat (current): ${preds['current']:.4f}")
print(f"1 saat sonrası tahmin  : ${preds['1h']:.4f}")
print(f"4 saat sonrası tahmin  : ${preds['4h']:.4f}")
print(f"24 saat sonrası tahmin : ${preds['24h']:.4f}")
