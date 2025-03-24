import pandas as pd
import numpy as np
import ta

# Descarga de datos
data = pd.read_csv('aapl_5m_test.csv').dropna()

rsi = ta.momentum.RSIIndicator(data.Close, window=23)
ultimate = ta.momentum.UltimateOscillator(high=data['High'], low=data['Low'], close=data['Close'], window1=7, window2=14, window3=28)
williams = ta.momentum.WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close'], lbp=14)

dataset = data.copy()

dataset['RSI'] = rsi.rsi()
dataset['Ultimate'] = ultimate.ultimate_oscillator()
dataset['Williams'] = williams.williams_r()

dataset['RSI_BUY'] = dataset['RSI'] < 15
dataset['RSI_SELL'] = dataset['RSI'] > 75

dataset['WILLIAMS_BUY'] = dataset['Williams'] < -80
dataset['WILLIAMS_SELL'] = dataset['Williams'] > -20

dataset['ULTIMATE_BUY'] = dataset['Ultimate'] < 30
dataset['ULTIMATE_SELL'] = dataset['Ultimate'] > 70

dataset = dataset.dropna()

print(dataset.head())