import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Veritabanı bağlantısı ve veri yükleme
conn = sqlite3.connect("btc_dataset.db")
price_df = pd.read_sql_query("SELECT * FROM ohlcv_1h", conn)
indicators_df = pd.read_sql_query("SELECT * FROM indicators_1h", conn)
conn.close()

# Veriyi birleştir ve temizle
df = pd.merge(indicators_df, price_df, on='timestamp', how='inner')
df = df.sort_values("timestamp").reset_index(drop=True)


# Gelişmiş veri temizleme fonksiyonu
def clean_data(df):
    # Sonsuz değerleri temizle
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Özellik mühendisliği (daha güvenli versiyon)
    df['log_return_1h'] = np.log1p(df['close'].pct_change().fillna(0)).clip(-0.5, 0.5)
    df['volume_change'] = df['volume'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

    # Teknik göstergeler
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['ema_diff'] = df['ema_12'] - df['ema_26']

    # Hedef değişkenler (daha güvenli hesaplama)
    for horizon in [1, 4, 24]:
        df[f'target_{horizon}h'] = df['close'].pct_change(horizon).shift(-horizon) * 100

    # Volatilite hesaplama
    df['volatility_24h'] = df['log_return_1h'].rolling(24, min_periods=1).std()

    # Son NaN kontrolü
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


df = clean_data(df)

# Özellik seçimi
features = ['rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'obv', 'ema',
            'log_return_1h', 'volatility_24h', 'volume_change',
            'ema_diff', 'open', 'high', 'low', 'close', 'volume']

targets = ['target_1h', 'target_4h', 'target_24h']

# RobustScaler ile ölçekleme (hata yönetimi ile)
try:
    scaler = RobustScaler()
    df[features] = scaler.fit_transform(df[features])
except Exception as e:
    print("Scaler hatası:", e)
    # Hangi özellikte sorun var?
    for col in features:
        if np.isinf(df[col]).any() or df[col].isna().any():
            print(f"Sorunlu sütun: {col}")
    raise


# Dataset sınıfı
class CryptoDataset(Dataset):
    def __init__(self, df, features, targets, seq_len=168):
        self.data = df[features].values.astype(np.float32)
        self.targets = df[targets].values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 24  # 24 saat sonrası için hedef

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x), torch.tensor(y)


# Veri yükleme
SEQ_LEN = 168
train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)

train_dataset = CryptoDataset(train_df, features, targets, SEQ_LEN)
val_dataset = CryptoDataset(val_df, features, targets, SEQ_LEN)

# Batch boyutunu GPU belleğine göre ayarla
batch_size = 32 if torch.cuda.is_available() else 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Transformer Modeli (optimize edilmiş)
class BTCTransformer(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.regressor(x.mean(dim=1))


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


# Cihaz ve model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BTCTransformer(num_features=len(features)).to(device)

# Optimizasyon
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
criterion = nn.HuberLoss()


# Eğitim döngüsü
def train_model(model, train_loader, val_loader, epochs=50):
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        epoch_losses = []
        all_preds = []
        all_targets = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.6f}, Val Loss: {val_loss:.6f}")


train_model(model, train_loader, val_loader)

# 1. Scaler'ı KAPAT (veya sadece 'close' hariç tüm özellikleri scale et)
scaler = RobustScaler()
scale_features = [f for f in features]  # 'close' hariç
df[scale_features] = scaler.fit_transform(df[scale_features])


# 2. Tahmin fonksiyonunu güncelle:
def predict(model, df, features, scaler, seq_len):
    model.eval()

    # 1. Son pencereyi al
    window = df.iloc[-seq_len:].copy()

    # 2. Sadece features kolonlarını al ve scale et
    scaled_window = scaler.transform(window[features])  # ✅ Burada hata düzeliyor

    # 3. Tensor'a çevir
    x = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)

    # 4. Tahmin
    with torch.no_grad():
        preds = model(x).cpu().numpy().flatten()

    # 5. Aktüel fiyatı al
    current_price = df['close'].iloc[-1]

    # 6. Tahmini fiyatları orijinal ölçeğe çevir
    return {
        'current': current_price,
        '1h': current_price * (1 + preds[0]),
        '4h': current_price * (1 + preds[1]),
        '24h': current_price * (1 + preds[2])
    }

def evaluate(model, val_loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).cpu().numpy()
            preds.append(y_pred)
            targets.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    print("R2 Score:", r2_score(targets, preds))
    print("MAE:", mean_absolute_error(targets, preds))
    print("RMSE:", np.sqrt(mean_squared_error(targets, preds)))
    print("Modelin ilk tahminleri:", preds[:5])  # Mantıklı değerler mi?
    print("Gerçek target'lar:", val_df[targets].values[:5])  # Modelden ne kadar uzak?


print(val_df[['target_1h', 'target_4h', 'target_24h']].head())

# --- EVALUATION METRICS ---
def calculate_metrics(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    def adjusted_r2(targets, preds):
        mean_target = np.mean(targets)
        ss_total = np.sum((targets - mean_target) ** 2)
        ss_res = np.sum((targets - preds) ** 2)
        return 1 - (ss_res / (ss_total + 1e-10))  # Küçük bir epsilon ekle

    print("Adjusted R2:", adjusted_r2(targets, preds))

    print("Modelin ilk tahminleri:", preds[:5])  # Mantıklı değerler mi?
    print("Gerçek target'lar:", targets[:5])  # Modelden ne kadar uzak?

    metrics = {
        'MSE': mean_squared_error(targets, preds),
        'MAE': mean_absolute_error(targets, preds),
        'R2': r2_score(targets, preds),
        'Adjusted_R2': adjusted_r2(targets, preds)
    }
    return metrics


# Metrikleri hesapla ve yazdır
print("\n--- TRAIN METRİKLERİ ---")
train_metrics = calculate_metrics(model, train_loader)
print(f"MSE: {train_metrics['MSE']:.6f}")
print(f"MAE: {train_metrics['MAE']:.6f}")
print(f"R2: {train_metrics['R2']:.4f}")

print("\n--- VALIDATION METRİKLERİ ---")
val_metrics = calculate_metrics(model, val_loader)
print(f"MSE: {val_metrics['MSE']:.6f}")
print(f"MAE: {val_metrics['MAE']:.6f}")
print(f"R2: {val_metrics['R2']:.4f}")

# --- TAHMIN VE SONUÇLAR ---
current_price = df['close'].iloc[-1]  # Direkt son gerçek fiyatı al

# Tahminleri al (yüzde değişimler olarak)
with torch.no_grad():
    window = df[features].iloc[-SEQ_LEN:].values
    scaled_window = scaler.transform(window)
    x = torch.tensor(scaled_window, dtype=torch.float32).unsqueeze(0).to(device)
    pred_percent = model(x).cpu().numpy().flatten()


import pandas as pd
import sqlite3

# # 1. Veritabanından HAM veriyi kontrol
conn = sqlite3.connect("btc_dataset.db")
raw_data = pd.read_sql_query("SELECT timestamp, close FROM ohlcv_1h ORDER BY timestamp DESC LIMIT 10", conn)
# conn.close()

# # 2. DataFrame'deki veriyi kontrol
# print("\n--- VERİTABANINDAN HAM VERİ (Son 10 Kayıt) ---")
# print(raw_data)
latest_raw_close = raw_data['close'].iloc[0]

# # 3. Mevcut df'deki close değerleri
# print("\n--- MEVCUT DF'DEKİ CLOSE (Son 5 Kayıt) ---")
# print(df[['timestamp', 'close']].tail(5))

# # 4. Scaler'ın close üzerindeki etkisi
# dummy = np.zeros((1, len(features)))
# dummy[0, features.index('close')] = df['close'].iloc[-1]  # Son close değeri
# scaled_value = scaler.transform(dummy)[0, features.index('close')]
# print(f"\nScaler Test: {df['close'].iloc[-1]} → {scaled_value} (scale edilmiş)")

# # 5. Model çıktılarını kontrol
# model.eval()
# window = df[features].iloc[-SEQ_LEN:].values
# x = torch.tensor(scaler.transform(window), dtype=torch.float32).unsqueeze(0).to(device)
# with torch.no_grad():
#     preds = model(x).cpu().numpy().flatten()
# print("\nModel Çıktıları (Yüzde Değişim):", preds)

# # 6. Manuel hesaplama
# current_price = df['close'].iloc[-1]
# print("\n--- MANUEL HESAPLAMA ---")
# print(f"Şuanki Fiyat: {current_price}")
# print(f"1h Tahmini: {current_price * (1 + preds[0])}")
# print(f"24h Tahmini: {current_price * (1 + preds[2])}")


# Yüzdeleri uygula
print("\n--- Tahminler ---")
print(f"Aktüel Fiyat (Ham Veri): ${latest_raw_close:.2f}")
print(f"1 Saat Tahmini: ${latest_raw_close * (1 + (pred_percent[0] / 10)):.2f} (%{(pred_percent[0] / 10)*100:.2f})")
print(f"4 Saat Tahmini: ${latest_raw_close * (1 + (pred_percent[1] / 10)):.2f} (%{(pred_percent[1] / 10)*100:.2f})")
print(f"24 Saat Tahmini: ${latest_raw_close * (1 + (pred_percent[2] / 10)):.2f} (%{(pred_percent[2] / 10)*100:.2f})")