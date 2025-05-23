import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths

import pandas as pd
import numpy as np
import ccxt
import ta  # pip install ta
from datetime import datetime
import sqlite3
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

conn = sqlite3.connect("btc_dataset.db")  # Dosya adını seninkine göre değiştir

price_df = pd.read_sql_query("SELECT * FROM ohlcv_1h", conn)
indicators_df = pd.read_sql_query("SELECT * FROM indicators_1h", conn)

# Merge işlemi
df = pd.merge(indicators_df, price_df, on='timestamp', how='inner')
df = df.sort_values("timestamp").reset_index(drop=True)

# Feature listesi
features = ['rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'obv', 'ema',
            'open', 'high', 'low', 'close', 'volume']

# Eksik verileri temizle
df.dropna(inplace=True)

# Normalize et
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

SEQ_LEN = 500  # 60 saatlik geçmişe bak
PRED_OFFSET = 24  # 1 saat sonrası hedef


class CryptoDataset(Dataset):
    def __init__(self, df, features):
        self.data = df[features].values.astype(np.float32)
        self.labels = df['close'].values.astype(np.float32)

    def __len__(self):
        return max(0, len(self.data) - SEQ_LEN - 24)  # max offset 24 saat

    def __getitem__(self, idx):
        x = self.data[idx:idx + SEQ_LEN]
        y_1h = self.labels[idx + SEQ_LEN + 1 - 1]     # 1 saat sonrası
        y_4h = self.labels[idx + SEQ_LEN + 4 - 1]     # 4 saat sonrası
        y_24h = self.labels[idx + SEQ_LEN + 24 - 1]   # 24 saat sonrası
        y = np.array([y_1h, y_4h, y_24h], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y)


dataset = CryptoDataset(df, features)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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
    
    
def multi_horizon_predict(model, df, features, scaler, seq_len):
    model.eval()
    window = df[features].values[-seq_len:].astype(np.float32)
    x = torch.tensor(window).unsqueeze(0).to(device)  # (1, seq_len, features)

    with torch.no_grad():
        y_pred_norm = model(x).cpu().numpy().flatten()

    # Ters ölçekleme
    dummy = np.zeros((1, len(features)))
    preds = {}
    for i, (label, norm_val) in enumerate(zip(["1h", "4h", "24h"], y_pred_norm)):
        dummy[0, features.index("close")] = norm_val
        preds[label] = scaler.inverse_transform(dummy)[0, features.index("close")]

    # Mevcut fiyatı da göster
    latest_close_scaled = df['close'].values[-1]
    dummy[0, features.index("close")] = latest_close_scaled
    preds["current"] = scaler.inverse_transform(dummy)[0, features.index("close")]
    return preds



# Model, loss ve optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerRegressor(num_features=len(features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Eğitim döngüsü
for epoch in range(10):
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
    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}")

preds = multi_horizon_predict(model, df, features, scaler, SEQ_LEN)

print(f"Aktüel fiyat  : ${preds['current']:.4f}")
print("--- Tahminler ---")
print(f"1 saat sonra  : ${preds['1h']:.4f}")
print(f"4 saat sonra  : ${preds['4h']:.4f}")
print(f"24 saat sonra : ${preds['24h']:.4f}")

