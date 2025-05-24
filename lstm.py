import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sqlite3
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri yükleme ve işleme kısmı (seninkiyle aynı)
conn = sqlite3.connect("btc_dataset.db")

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
scaled_features = scaler.fit_transform(df[features])
df[features] = scaled_features  # sadece normalize edilmiş veriyi tut

scaled_data = scaled_features  # eğitim için kullanacağın veri

close_idx = features.index('close')

# 3 farklı hedef: 1 saat, 4 saat, 24 saat sonrası close farkları
offsets = [1, 4, 24]  # saat cinsinden (veri 1 saatlik)

X, y = [], []
seq_len = 200

for i in range(seq_len, len(scaled_data) - max(offsets)):
    X.append(scaled_data[i - seq_len:i])
    targets = []
    for off in offsets:
        delta = scaled_data[i + off][close_idx] - scaled_data[i - 1][close_idx]
        targets.append(delta)
    y.append(targets)

X = np.array(X)
y = np.array(y)  # shape: (num_samples, 3)

# Train/val split
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Tensorlara dönüştür
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)  # shape (N, 3)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# Model: çıkışı 3 boyutlu (3 zaman farkı)
class MultiOutputLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=3):
        super(MultiOutputLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        out = self.fc(x)
        return out

model = MultiOutputLSTMModel(input_size=len(features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Eğitim (örnek 10 epoch)
epochs = 10
batch_size = 8

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / (X_train.size(0) // batch_size)

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, X_val.size(0), batch_size):
            batch_x = X_val[i:i+batch_size]
            batch_y = y_val[i:i+batch_size]
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    avg_val_loss = val_loss / (X_val.size(0) // batch_size)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Validation Loss: {avg_val_loss:.6f}")

# Eğitim sonrası son pencere için tahminler

last_seq = scaled_data[-seq_len:]  # (seq_len, features)
last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    pred_deltas = model(last_seq_tensor).cpu().numpy().flatten()  # [delta_1h, delta_4h, delta_24h]

last_close_scaled = scaled_data[-1, close_idx]

predicted_prices = []
for delta in pred_deltas:
    pred_scaled = last_close_scaled + delta
    dummy = np.zeros((1, len(features)))
    dummy[0, close_idx] = pred_scaled
    price = scaler.inverse_transform(dummy)[0, close_idx]
    predicted_prices.append(price)

current_price = scaler.inverse_transform(np.expand_dims(scaled_data[-1], axis=0))[0, close_idx]

print(f"Şu anki fiyat: {current_price:.2f}")
print(f"1 saat sonrası tahmini fiyat: {predicted_prices[0]:.2f}")
print(f"4 saat sonrası tahmini fiyat: {predicted_prices[1]:.2f}")
print(f"24 saat sonrası tahmini fiyat: {predicted_prices[2]:.2f}")

# Validation set için tahminleri batch batch al
model.eval()
batch_size = 64
preds_delta_list = []

with torch.no_grad():
    for i in range(0, X_val.size(0), batch_size):
        batch_x = X_val[i:i+batch_size]
        preds = model(batch_x).cpu().numpy()
        preds_delta_list.append(preds)

preds_delta = np.vstack(preds_delta_list)  # shape: (num_val_samples, 3)
y_val_np = y_val.cpu().numpy()
X_val_np = X_val.cpu().numpy()

close_idx = features.index('close')
last_close_prices = X_val_np[:, -1, close_idx]  # shape (num_val_samples,)

# Her hedef için ayrı tahmin edilen fiyatları hesapla
predicted_prices_list = []
real_prices_list = []

for target_idx in range(len(offsets)):  # 0:1h, 1:4h, 2:24h
    preds_for_target = last_close_prices + preds_delta[:, target_idx]
    real_for_target = last_close_prices + y_val_np[:, target_idx]

    predicted_prices_list.append(preds_for_target)
    real_prices_list.append(real_for_target)

# R2 skorlarını hedef bazında hesapla
from sklearn.metrics import r2_score

for i, off in enumerate(offsets):
    r2 = r2_score(real_prices_list[i], predicted_prices_list[i])
    print(f"Validation R2 Score for {off} hour(s) ahead (scaled): {r2:.4f}")

# İstersen inverse scale edip gerçek fiyatlara da dönebilirsin
for i, off in enumerate(offsets):
    dummy_pred = np.zeros((len(predicted_prices_list[i]), len(features)))
    dummy_pred[:, close_idx] = predicted_prices_list[i]
    dummy_real = np.zeros((len(real_prices_list[i]), len(features)))
    dummy_real[:, close_idx] = real_prices_list[i]

    inv_pred_prices = scaler.inverse_transform(dummy_pred)[:, close_idx]
    inv_real_prices = scaler.inverse_transform(dummy_real)[:, close_idx]

    r2_inv = r2_score(inv_real_prices, inv_pred_prices)
    print(f"Validation R2 Score for {off} hour(s) ahead (inverse scaled): {r2_inv:.4f}")

# İstersen ilk 10 tahmin ve gerçek değerleri göster (inverse scaled)
print("\nİlk 10 Validation Tahmini ve Gerçek Değerler (inverse scaled):")
for i in range(10):
    print(f"Gerçek 1h: {inv_real_prices[i]:.2f} - Tahmin 1h: {inv_pred_prices[i]:.2f}")