from indicators import trading_signals


dataset_with_signals, trained_model = trading_signals(
    filepath='aapl_5m_train.csv',
    plot_signals=True
)