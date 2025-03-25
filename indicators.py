import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def trading_signals(filepath='aapl_5m_train.csv', plot_signals=True):
    """
    Carga datos, calcula indicadores técnicos, entrena un SVM y genera señales de trading.
    Retorna el dataset con las señales incluidas y el modelo entrenado.
    """

    # Cargar y preparar datos
    data = pd.read_csv(filepath)
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)

    # Hacer copia para no modificar el original
    dataset = data.copy()

    # Indicadores técnicos
    rsi = ta.momentum.RSIIndicator(dataset.Close, window=23)
    ultimate = ta.momentum.UltimateOscillator(
        high=dataset['High'], low=dataset['Low'], close=dataset['Close'],
        window1=7, window2=14, window3=28
    )
    williams = ta.momentum.WilliamsRIndicator(
        high=dataset['High'], low=dataset['Low'], close=dataset['Close'], lbp=14
    )

    dataset['RSI'] = rsi.rsi()
    dataset['Ultimate'] = ultimate.ultimate_oscillator()
    dataset['Williams'] = williams.williams_r()

    # Retorno futuro a 2 horas
    dataset['future_return'] = dataset['Close'].shift(-24) / dataset['Close'] - 1

    buy_threshold = 0.0015
    sell_threshold = -0.0015

    def generate_signal(x):
        if x > buy_threshold:
            return 'BUY'
        elif x < sell_threshold:
            return 'SELL'
        else:
            return 'WAIT'

    dataset['signal'] = dataset['future_return'].apply(generate_signal)
    dataset = dataset.dropna(subset=['RSI', 'Ultimate', 'Williams', 'signal'])

    # Features, etiquetas e índices
    features = ['RSI', 'Ultimate', 'Williams']
    X = dataset[features]
    y = dataset['signal']
    close_prices = dataset['Close']
    index = dataset.index

    # División temporal
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, index, test_size=0.2, shuffle=False
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modelo SVM
    svm = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')
    svm.fit(X_train_scaled, y_train)

    # Predicción
    y_pred = svm.predict(X_test_scaled)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Asignar predicciones al dataset original
    dataset.loc[idx_test, 'predicted_signal'] = y_pred

    # Gráfica de señales
    if plot_signals:
        test_plot_df = pd.DataFrame({
            'Close': close_prices.loc[idx_test].values,
            'Prediction': y_pred
        }, index=idx_test)

        plt.figure(figsize=(12,6))
        plt.plot(test_plot_df['Close'], label='Close Price', alpha=0.6)

        plt.scatter(test_plot_df.index[test_plot_df['Prediction'] == 'BUY'],
                    test_plot_df['Close'][test_plot_df['Prediction'] == 'BUY'],
                    marker='^', color='green', label='BUY')

        plt.scatter(test_plot_df.index[test_plot_df['Prediction'] == 'SELL'],
                    test_plot_df['Close'][test_plot_df['Prediction'] == 'SELL'],
                    marker='v', color='red', label='SELL')

        plt.legend()
        plt.title("SVM Trading Signals")
        plt.xlabel("Datetime")
        plt.ylabel("Price")
        plt.grid(True)
        plt.show()

    return dataset, svm