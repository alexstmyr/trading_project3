from indicators import train_and_predict_svm_signals
from backtesting import run_backtest
import matplotlib.pyplot as plt

# Entrenar modelo y generar señales
dataset, model = train_and_predict_svm_signals('aapl_5m_train.csv', plot_signals=False)

# Ejecutar backtest
portfolio_value, final_capital = run_backtest(dataset)

# Mostrar resultados
print(f"Capital final: ${final_capital:,.2f}")

# Graficar evolución
plt.figure(figsize=(12, 6))
plt.plot(portfolio_value, label='Portfolio Value')
plt.title("Evolución del portafolio durante el backtest")
plt.xlabel("Pasos de tiempo")
plt.ylabel("Valor del portafolio")
plt.grid(True)
plt.legend()
plt.show()