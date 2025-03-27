import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn
from sklearn.metrics import f1_score

class TechnicalIndicator:
    
    def __init__(self, name: str, function):
        self.name = name
        self.function = function
        
        
        
class Dataset:
    
    def __init__(self, dataframe: pd.DataFrame = pd.DataFrame()):
        self.dataframe = dataframe
        self.indicators = {}
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        
    def add_indicator(self, indicator: TechnicalIndicator = None):
        self.indicators[indicator.name] = indicator
        self.dataframe[indicator.name] = 0
    
    
    def lag_data(self, column: str = "Close", lags: int = 1):
        for lag in range (1, lags +1):
            self.dataframe["lag_{}".format(lag)] = self.dataframe[column].shift(lag)
    
    
    def generate_signal(self, buy_threshold: float = 0.015, sell_threshold: float = -0.015, lag: int = -24):
        self.dataframe["Return"] = self.dataframe["Close"].shift(lag) / self.dataframe["Close"] - 1
        
        self.dataframe["Signal"] = "WAIT"
        self.dataframe.loc[self.dataframe["Return"] > buy_threshold, "Signal"] = "BUY"
        self.dataframe.loc[self.dataframe["Return"] < sell_threshold, "Signal"] = "SELL"
  
    
    def define_features_target(self, features: list, target: str):
        self.X = self.dataframe[features]
        self.y = self.dataframe[target]
        
        
    def scale_data(self, fit: bool = True, scaler: StandardScaler = StandardScaler()):
        if fit:
            scaled = scaler.fit_transform(self.X) 
        else:
            scaled = scaler.transform(self.X) 
        self.X = pd.DataFrame(scaled, columns=self.X.columns, index=self.X.index)
    
    
    @staticmethod
    def drop_columns(dataframe: pd.DataFrame, columns: list = None):
        pass
    

class Model:
    
    def __init__(self, svm: sklearn.svm._classes.SVC = SVC(), kernel: str = "rbf", gamma: str = "scale", weight: str = "balanced", iter: int = 10_000):
        self.svm = svm
        self.svm.kernel = kernel
        self.svm.gamma = gamma
        self.svm.class_weight = weight
        self.svm.max_iter = iter
        self.ypred = None
        self.f1 = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.svm.fit(X, y)
        
    def predict(self, X: pd.DataFrame):
        self.ypred = self.svm.predict(X)
        
    def f1_score(self, test: pd.Series, pred: pd.Series, average: str = "macro"):
        self.f1 = f1_score(test, pred, average=average)
        
class Backtest:
    
    def __init__(self, capital: int = 1_000_000, n_shares: int = 2_000, com: float = 0.125 / 100, stop_loss: float = 0.1, take_profit: float = 0.1):
        self.capital = capital
        self.portfolio_value = [capital]
        self.active_long_pos = None
        self.active_short_pos = None
        self.n_shares = n_shares
        self.com = com
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.calmar_ratio = 0
        
    def calculate_portfolio(self, dataset: pd.DataFrame, capital: int = 1_000_000):
        
        self.capital = capital
        self.active_long_pos = None
        self.active_short_pos = None
        self.portfolio_value = [self.capital]
        
        for i, row in dataset.iterrows():
            
            # Cerrar posición Long activa
            if self.active_long_pos:
                # Cierre por stop loss or take profit
                if row.Close < self.active_long_pos['stop_loss'] or row.Close > self.active_long_pos['take_profit']:
                    pnl = row.Close * self.n_shares * (1 - self.com)
                    self.capital += pnl
                    self.active_long_pos = None
            
            # Cerrar posición Short activa
            if self.active_short_pos:
                if row.Close > self.active_short_pos['stop_loss'] or row.Close < self.active_short_pos['take_profit']:
                    # Recomprar caro = pérdida
                    cost = row.Close * self.n_shares * (1 + self.com)
                    pnl = self.active_short_pos['entry'] * self.n_shares - cost
                    self.capital += pnl  # Restamos pérdida
                    self.active_short_pos = None
                

            # Abrir posición Long
            if row["Signal"] == 'BUY' and self.active_long_pos is None:
                cost = row.Close * self.n_shares * (1 + self.com)
                if self.capital > cost:
                    self.capital -= cost
                    self.active_long_pos = {
                        'datetime': row.name,
                        'opened_at': row.Close,
                        'take_profit': row.Close * (1 + self.take_profit),
                        'stop_loss': row.Close * (1 - self.stop_loss)
                    }

            # Abrir posición Short (solo si no hay posición Long activa)
            if row["Signal"] == 'SELL' and self.active_short_pos is None and self.active_long_pos is None:
                proceeds = row.Close * self.n_shares * (1 - self.com)
                self.capital += proceeds
                self.active_short_pos = {
                    'datetime': row.name,
                    'entry': row.Close,
                    'take_profit': row.Close * (1 - self.take_profit),
                    'stop_loss': row.Close * (1 + self.stop_loss)
                }

            # Calcular el valor de la posición actual
            position_value = 0
            if self.active_long_pos:
                position_value = row.Close * self.n_shares
            elif self.active_short_pos:
                position_value = self.active_short_pos['entry'] * self.n_shares - row.Close * self.n_shares

            # Actualizar el valor total del portafolio
            self.portfolio_value.append(self.capital + position_value)
    
    def calculate_calmar(self, bars_per_year = 20_280):
        
        initial_val = self.portfolio_value[0]
        final_val = self.portfolio_value[-1]
        n_bars = len(self.portfolio_value)
        
        # CAGR
        if initial_val <= 0 or final_val <= 0:
            return 0.0  # Evita divisiones por cero o valores no válidos
        
        cagr = (final_val / initial_val) ** (bars_per_year / n_bars) - 1
        
        # Max Drawdown
        # Para calcular MDD, podemos hacer un track del máximo acumulado y ver la caída relativa
        max_so_far = self.portfolio_value[0]
        max_drawdown = 0
        for pv in self.portfolio_value:
            if pv > max_so_far:
                max_so_far = pv
            drawdown = (max_so_far - pv) / max_so_far
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        if max_drawdown == 0:
            # Si nunca hubo drawdown, el ratio sería infinito;
            # Para que no rompa, devolvemos el CAGR en lugar de infinito.
            return cagr if cagr > 0 else 0.0
        
        self.calmar_ratio = cagr / max_drawdown