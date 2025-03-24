import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


# Descarga de datos
data = pd.read_csv('aapl_5m_test.csv').dropna()

rsi = ta.momentum.RSIIndicator(data.Close, window=23)
ultimate = ta.momentum.UltimateOscillator(high=data['High'], low=data['Low'], close=data['Close'], window1=7, window2=14, window3=28)
williams = ta.momentum.WilliamsRIndicator(high=data['High'], low=data['Low'], close=data['Close'], lbp=14)

dataset = data.copy()

# Agregar indicadores técnicos al dataset
dataset['RSI'] = rsi.rsi()
dataset['Ultimate'] = ultimate.ultimate_oscillator()
dataset['Williams'] = williams.williams_r()

dataset = dataset.dropna()

# Retornos de cada 2 horas
dataset['future_return'] = dataset['Close'].shift(-24) / dataset['Close'] - 1

# Establecer umbrales para las señales
buy_threshold = 0.003  
sell_threshold = -0.003

def generate_signal(x):
    if x > buy_threshold:
        return 'BUY'
    elif x < sell_threshold:
        return 'SELL'
    else:
        return 'WAIT'

dataset['signal'] = dataset['future_return'].apply(generate_signal)

dataset['Datetime'] = pd.to_datetime(dataset['Datetime'])
dataset.set_index('Datetime', inplace=True)

# Seleccionar solo cada 2 horas
dataset_2h = dataset.iloc[::24, :]

features = ['RSI', 'Ultimate', 'Williams']
X = dataset_2h[features]
y = dataset_2h['signal']

# Entrenar el modelo SVM

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Escalado para mejor funcionamiento del modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar SVM
svm = SVC(kernel='rbf', C=1, gamma='scale')
svm.fit(X_train_scaled, y_train)

# Predicciones
y_pred = svm.predict(X_test_scaled)
print(classification_report(y_test, y_pred))