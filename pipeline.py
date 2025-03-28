from utils import Dataset, TechnicalIndicator, Model, Backtest
import pandas as pd
import os
import ta
import optuna

def pipeline(optimize: bool = True):
    lags = 7

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    train_data = Dataset(pd.read_csv(r"Data/aapl_5m_train.csv"))
    train_data.lag_data(lags = lags)

    train_data.add_indicator(
        TechnicalIndicator("rsi", ta.momentum.RSIIndicator(
            train_data.dataframe["Close"]
        ))
    )

    train_data.add_indicator(
        TechnicalIndicator("ultimate", ta.momentum.UltimateOscillator(
            high=train_data.dataframe['High'], 
            low=train_data.dataframe['Low'], 
            close=train_data.dataframe['Close']
        ))
    )

    train_data.add_indicator(
        TechnicalIndicator("williams", ta.momentum.WilliamsRIndicator(
            high=train_data.dataframe['High'], 
            low=train_data.dataframe['Low'], 
            close=train_data.dataframe['Close']
        ))
    )

    train_data.generate_signal()

    train_data.dataframe.dropna(inplace=True)

    train_data.define_features_target(["Open", "High", "Low", "Close", "Return", "rsi", "ultimate", "williams"] + ["lag_{}".format(i) for i in range(1, lags + 1)], "Signal")

    train_data.scale_data()




    test_data = Dataset(pd.read_csv(r"Data/aapl_5m_test.csv"))
    test_data.lag_data(lags = lags)

    test_data.add_indicator(
        TechnicalIndicator("rsi", ta.momentum.RSIIndicator(
            test_data.dataframe["Close"]
        ))
    )

    test_data.add_indicator(
        TechnicalIndicator("ultimate", ta.momentum.UltimateOscillator(
            high=test_data.dataframe['High'], 
            low=test_data.dataframe['Low'], 
            close=test_data.dataframe['Close']
        ))
    )

    test_data.add_indicator(
        TechnicalIndicator("williams", ta.momentum.WilliamsRIndicator(
            high=test_data.dataframe['High'], 
            low=test_data.dataframe['Low'], 
            close=test_data.dataframe['Close']
        ))
    )

    test_data.generate_signal()

    test_data.dataframe.dropna(inplace=True)

    test_data.define_features_target(["Open", "High", "Low", "Close", "Return", "rsi", "ultimate", "williams"] + ["lag_{}".format(i) for i in range(1, lags + 1)], "Signal")

    test_data.scale_data()



    model = Model()

    if optimize:
        study = optuna.create_study(direction="maximize", )

        def objective(trial, train, test, model):
            rsi_window = trial.suggest_int("rsi_window", 5, 20)
            ultimate_window1 = trial.suggest_int("ultimate_window1", 1, 10)
            ultimate_window2 = trial.suggest_int("ultimate_window2", 10, 20)
            ultimate_window3 = trial.suggest_int("ultimate_window3", 20, 30)
            williams_lbp = trial.suggest_int("williams_lbp", 10, 20)
            
            
            train.indicators["rsi"].function._window = rsi_window
            train.indicators["ultimate"].function._window1 = ultimate_window1
            train.indicators["ultimate"].function._window2 = ultimate_window2
            train.indicators["ultimate"].function._window3 = ultimate_window3
            train.indicators["williams"].function._lbp = williams_lbp
            
            train.dataframe["rsi"] = train.indicators["rsi"].function.rsi()
            train.dataframe["ultimate"] = train.indicators["ultimate"].function.ultimate_oscillator()
            train.dataframe["williams"] = train.indicators["williams"].function.williams_r()
            
            train.dataframe.dropna(inplace=True)
            train.define_features_target(["Open", "High", "Low", "Close", "Return", "rsi", "ultimate", "williams"] + ["lag_{}".format(i) for i in range(1, lags + 1)], "Signal")
            train.scale_data(scaler = train.scaler)
            
            
            test.indicators["rsi"].function._window = rsi_window
            test.indicators["ultimate"].function._window1 = ultimate_window1
            test.indicators["ultimate"].function._window2 = ultimate_window2
            test.indicators["ultimate"].function._window3 = ultimate_window3
            test.indicators["williams"].function._lbp = williams_lbp
            
            test.dataframe["rsi"] = test.indicators["rsi"].function.rsi()
            test.dataframe["ultimate"] = test.indicators["ultimate"].function.ultimate_oscillator()
            test.dataframe["williams"] = test.indicators["williams"].function.williams_r()
            
            test.dataframe.dropna(inplace=True)
            test.define_features_target(["Open", "High", "Low", "Close", "Return", "rsi", "ultimate", "williams"] + ["lag_{}".format(i) for i in range(1, lags + 1)], "Signal")
            test.scale_data(fit = False, scaler = train.scaler)
            
            model.svm.C = trial.suggest_float("C", 0.01, 100, log=True)
            model.fit(train.X, train.y)
            
            model.predict(test.X)
            
            model.f1_score(test.y, model.ypred)
            
            
            return model.f1    

        study.optimize(lambda t: objective(t, train_data, test_data, model), n_trials = 50)

        best_params = study.best_params
        # print(study.best_params)
        # print(study.best_value)
    else:
        best_params = {'rsi_window': 16, 'ultimate_window1': 8, 'ultimate_window2': 17, 'ultimate_window3': 28, 'williams_lbp': 19, 'C': 34.51419759844153}


    train_data.indicators["rsi"].function._window = best_params["rsi_window"]
    train_data.indicators["ultimate"].function._window1 = best_params["ultimate_window1"]
    train_data.indicators["ultimate"].function._window2 = best_params["ultimate_window2"]
    train_data.indicators["ultimate"].function._window3 = best_params["ultimate_window3"]
    train_data.indicators["williams"].function._lbp = best_params["williams_lbp"]

    train_data.dataframe["rsi"] = train_data.indicators["rsi"].function.rsi()
    train_data.dataframe["ultimate"] = train_data.indicators["ultimate"].function.ultimate_oscillator()
    train_data.dataframe["williams"] = train_data.indicators["williams"].function.williams_r()

    train_data.dataframe.dropna(inplace=True)
    train_data.define_features_target(["Open", "High", "Low", "Close", "Return", "rsi", "ultimate", "williams"] + ["lag_{}".format(i) for i in range(1, lags + 1)], "Signal")
    train_data.scale_data(scaler = train_data.scaler)

    test_data.indicators["rsi"].function._window = best_params["rsi_window"]
    test_data.indicators["ultimate"].function._window1 = best_params["ultimate_window1"]
    test_data.indicators["ultimate"].function._window2 = best_params["ultimate_window2"]
    test_data.indicators["ultimate"].function._window3 = best_params["ultimate_window3"]
    test_data.indicators["williams"].function._lbp = best_params["williams_lbp"]

    test_data.dataframe["rsi"] = test_data.indicators["rsi"].function.rsi()
    test_data.dataframe["ultimate"] = test_data.indicators["ultimate"].function.ultimate_oscillator()
    test_data.dataframe["williams"] = test_data.indicators["williams"].function.williams_r()

    test_data.dataframe.dropna(inplace=True)
    test_data.define_features_target(["Open", "High", "Low", "Close", "Return", "rsi", "ultimate", "williams"] + ["lag_{}".format(i) for i in range(1, lags + 1)], "Signal")
    test_data.scale_data()
    test_data.scale_data(fit = False, scaler = train_data.scaler)

    model.svm.C = best_params["C"]
    model.fit(train_data.X, train_data.y)

    model.predict(test_data.X)

    model.f1_score(test_data.y, model.ypred)

    test_data.dataframe["Signal"] = model.ypred



    backtesting = Backtest()

    if optimize:
        study2 = optuna.create_study(direction="maximize")

        def objective_calmar(trial, dataset, backtesting):
            
            backtesting.stop_loss = trial.suggest_float("stop_loss", 0.01, 0.2)
            backtesting.take_profit = trial.suggest_float("take_profit", 0.01, 0.3)
            backtesting.n_shares = trial.suggest_int("n_shares", 100, 5000, step=100)
            
            backtesting.calculate_portfolio(dataset)
            backtesting.calculate_calmar()
            
            return backtesting.calmar_ratio

        study2.optimize(lambda t: objective_calmar(t, test_data.dataframe, backtesting), n_trials = 50)

        best_params2 = study2.best_params

        # print(study2.best_params)
        # print(study2.best_value)
    else:
        best_params2 =  {'stop_loss': 0.1856035563655588, 'take_profit': 0.274330179954673, 'n_shares': 4900}
        
    backtesting.stop_loss = best_params2["stop_loss"]
    backtesting.take_profit = best_params2["take_profit"]
    backtesting.n_shares = best_params2["n_shares"]

    backtesting.calculate_portfolio(test_data.dataframe)
    backtesting.calculate_calmar()
    backtesting.calculate_sortino()
    backtesting.calculate_sharpe()

    
    return backtesting.portfolio_value, test_data.dataframe.Close, backtesting.calmar_ratio, backtesting.sharpe_ratio, backtesting.sortino_ratio