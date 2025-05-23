import sqlite3
import pandas as pd
from binance.client import Client
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import AverageTrueRange
from dotenv import load_dotenv
import time
import os

load_dotenv()
# Enter your own API keys
API_KEY = os.environ['BINANCE_TESTNET_API_KEY']
API_SECRET = os.environ.get('BINANCE_TESTNET_API_SECRET')
client = Client(API_KEY, API_SECRET)

# Enter a sybol to download data
symbol = 'BTCUSDT'
intervals = {
    '5m': Client.KLINE_INTERVAL_5MINUTE,
    '15m': Client.KLINE_INTERVAL_15MINUTE,
    '30m': Client.KLINE_INTERVAL_30MINUTE,
    '1h': Client.KLINE_INTERVAL_1HOUR
}

# Name Dataset
conn = sqlite3.connect('btc_dataset.db')
cursor = conn.cursor()

# Tables
for name in intervals.keys():
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS ohlcv_{name} (
            timestamp INTEGER PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
        )
    ''')
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS indicators_{name} (
            timestamp INTEGER PRIMARY KEY,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            bb_upper REAL,
            bb_middle REAL,
            bb_lower REAL,
            atr REAL,
            obv REAL,
            ema REAL
        )
    ''')
conn.commit()


# Indicator Calculations
def compute_indicators(df):
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    bb = BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['ema'] = EMAIndicator(df['close'], window=20).ema_indicator()
    return df


def fetch_and_store_data(interval_name, binance_interval):
    print(f"Fetching {interval_name} data...")

    # You can change date to fetch more or less data
    klines = client.get_historical_klines(symbol, binance_interval, start_str='1 Jan 2020')

    data = []
    for k in klines:
        timestamp = int(k[0])
        open_ = float(k[1])
        high = float(k[2])
        low = float(k[3])
        close = float(k[4])
        volume = float(k[5])
        data.append([timestamp, open_, high, low, close, volume])

    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = compute_indicators(df)

    for _, row in df.iterrows():
        cursor.execute(f'''
            INSERT OR IGNORE INTO ohlcv_{interval_name} (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (int(row['timestamp']), row['open'], row['high'], row['low'], row['close'], row['volume']))

        cursor.execute(f'''
            INSERT OR IGNORE INTO indicators_{interval_name}
            (timestamp, rsi, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower, atr, obv, ema)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            int(row['timestamp']),
            row['rsi'], row['macd'], row['macd_signal'], row['macd_hist'],
            row['bb_upper'], row['bb_middle'], row['bb_lower'],
            row['atr'], row['obv'], row['ema']
        ))

    conn.commit()
    print(f"{interval_name} data saved. Data count: {len(df)}")


for name, binance_interval in intervals.items():
    fetch_and_store_data(name, binance_interval)
    time.sleep(1)  # Rate limit protection

print("All data saved successfully!")
