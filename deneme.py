import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sqlite3
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri yükleme ve ön işleme
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

# Train/val split öncesi scaler uygulama (data leakage önlemek için)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
val_df = df.iloc[split_idx:].copy()

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[features])
val_scaled = scaler.transform(val_df[features])

# Orijinal close fiyatlarını sakla (inverse transform için)
train_close_prices = train_df['close'].values
val_close_prices = val_df['close'].values

# Sequence ve target oluşturma
seq_len = 336  # 2 haftalık veri (1h candles)
offsets = [1, 4, 24]  # 1h, 4h, 24h sonrası


def create_sequences(data, close_prices, seq_len, offsets):
    X, y = [], []
    close_idx = features.index('close')

    for i in range(seq_len, len(data) - max(offsets)):
        X.append(data[i - seq_len:i])

        # Yüzde değişim olarak hedef
        current_close = data[i - 1, close_idx]
        targets = []
        for off in offsets:
            future_close = data[i + off, close_idx]
            delta = (future_close - current_close) / current_close  # yüzde değişim
            targets.append(delta)
        y.append(targets)

    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_scaled, train_close_prices, seq_len, offsets)
X_val, y_val = create_sequences(val_scaled, val_close_prices, seq_len, offsets)

# PyTorch tensörlerine dönüştürme
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
y_val = torch.FloatTensor(y_val).to(device)


# Gelişmiş Model Mimarisi
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, output_size=3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(hidden_size1 * 2)
        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x.permute(0, 2, 1)
        x = self.bn1(x).permute(0, 2, 1)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.bn2(x[:, -1, :])
        x = self.dropout2(x)

        return self.fc(x)


model = EnhancedLSTMModel(input_size=len(features)).to(device)
criterion = nn.SmoothL1Loss()  # MSE yerine
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Eğitim döngüsü
batch_size = 64
epochs = 30

best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    # Random batching
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)

        # Ağırlıklı multi-task loss
        loss = 0.3 * criterion(outputs[:, 0], batch_y[:, 0]) + \
               0.3 * criterion(outputs[:, 1], batch_y[:, 1]) + \
               0.4 * criterion(outputs[:, 2], batch_y[:, 2])

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / (X_train.size(0) // batch_size)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i in range(0, X_val.size(0), batch_size):
            batch_x, batch_y = X_val[i:i + batch_size], y_val[i:i + batch_size]
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

    avg_val_loss = val_loss / (X_val.size(0) // batch_size)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)

    # En iyi modeli kaydet
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    print(
        f"Epoch {epoch + 1}/{epochs} - Train: {avg_train_loss:.6f} - Val: {avg_val_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

# Loss grafiği
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.show()

# En iyi modeli yükle
model.load_state_dict(torch.load('best_model.pth'))


# Tahmin fonksiyonu
def predict_future(model, last_sequence, steps_ahead):
    model.eval()
    predictions = []
    current_seq = last_sequence.copy()

    with torch.no_grad():
        for _ in range(steps_ahead):
            input_tensor = torch.FloatTensor(current_seq[-seq_len:]).unsqueeze(0).to(device)
            pred_deltas = model(input_tensor).cpu().numpy()[0]

            # Son close değerini al
            last_close = current_seq[-1, features.index('close')]

            # Tahmin edilen yüzde değişimleri uygula
            new_close = last_close * (1 + pred_deltas[0])  # 1h sonrası için
            new_row = current_seq[-1].copy()
            new_row[features.index('close')] = new_close

            current_seq = np.vstack([current_seq, new_row])
            predictions.append(new_close)

    return predictions


# Son sequence'i al
last_seq = val_scaled[-seq_len:]

# 24 saatlik tahmin
future_predictions = predict_future(model, last_seq, 24)

# Inverse transform
dummy = np.zeros((len(future_predictions), len(features)))
dummy[:, features.index('close')] = future_predictions
predicted_prices = scaler.inverse_transform(dummy)[:, features.index('close')]

# Gerçek ve tahmin edilen değerleri görselleştirme
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices, label='Predicted')
plt.title('24 Hour Price Prediction')
plt.legend()
plt.show()

# Validation R2 hesaplama
model.eval()
with torch.no_grad():
    val_preds = model(X_val).cpu().numpy()

# Yüzde değişimden gerçek fiyata dönüşüm
close_idx = features.index('close')
val_last_closes = X_val[:, -1, close_idx].cpu().numpy()

# Her zaman periyodu için R2
for i, off in enumerate(offsets):
    # Tahmin edilen yüzde değişimleri uygula
    pred_prices_scaled = val_last_closes * (1 + val_preds[:, i])

    # Gerçek yüzde değişimleri uygula
    true_prices_scaled = val_last_closes * (1 + y_val[:, i].cpu().numpy())

    # Inverse scaling
    dummy_pred = np.zeros((len(pred_prices_scaled), len(features)))
    dummy_pred[:, close_idx] = pred_prices_scaled
    pred_prices = scaler.inverse_transform(dummy_pred)[:, close_idx]

    dummy_true = np.zeros((len(true_prices_scaled), len(features)))
    dummy_true[:, close_idx] = true_prices_scaled
    true_prices = scaler.inverse_transform(dummy_true)[:, close_idx]

    r2 = r2_score(true_prices, pred_prices)
    print(f"{off}h R2 Score: {r2:.4f}")

# Örnek tahminler
print("\nSample Predictions:")
for i in range(5):
    print(f"True 1h: {true_prices[i]:.2f} - Pred 1h: {pred_prices[i]:.2f}")