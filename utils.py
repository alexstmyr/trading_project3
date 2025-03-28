import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn
from sklearn.metrics import f1_score
import numpy as np

class TechnicalIndicator:
    """
    A technical indicator used in stock market analysis to generate trading signals.
    
    Attributes:
        name (str): The name of the technical indicator (e.g., 'rsi', 'macd').
        function: A technical indicator function from the 'ta' library that calculates 
                 the indicator values. The function should be callable and return 
                 a pandas Series when executed.
    """
    
    def __init__(self, name: str, function):
        """
        Initializes the TechnicalIndicator with a name and calculation function.
        
        Args:
            name: The name of the indicator.
            function: A callable that calculates the indicator values.
        """
        self.name = name
        self.function = function
        
        
class Dataset:
    """
    A dataset container for machine learning with financial data, specifically designed
    to work with technical indicators as features.
    
    Attributes:
        dataframe (pd.DataFrame): The main DataFrame containing all market data and features. Default is an empty DataFrame.
        indicators (dict): Dictionary of TechnicalIndicator objects used as features.
        X (pd.DataFrame): Design matrix containing selected features.
        y (pd.Series): Target variable series.
        scaler (StandardScaler): Scaler object for feature standardization.
    """
    
    def __init__(self, dataframe: pd.DataFrame = pd.DataFrame()):
        self.dataframe = dataframe
        self.indicators = {}
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        
    def add_indicator(self, indicator: TechnicalIndicator = None):
        """
        Adds a technical indicator to both the indicators dictionary and as a new column
        in the DataFrame.
        
        Args:
            indicator: A TechnicalIndicator object to be added.
        """
        self.indicators[indicator.name] = indicator
        self.dataframe[indicator.name] = 0
    
    def lag_data(self, column: str = "Close", lags: int = 1):
        """
        Creates lagged versions of a specified column.
        
        Args:
            column: Name of the column to lag. Default is 'Close'.
            lags: Number of lagged versions to create. Default is 1.
        """
        for lag in range (1, lags +1):
            self.dataframe["lag_{}".format(lag)] = self.dataframe[column].shift(lag)
    
    def generate_signal(self, buy_threshold: float = 0.015, sell_threshold: float = -0.015, lag: int = -24):
        """
        Generates trading signals (BUY/SELL/WAIT) based on future returns.
        
        Args:
            buy_threshold: Minimum return threshold for BUY signal. Default is 1.5%.
            sell_threshold: Maximum return threshold for SELL signal. Default is -1.5%.
            lag: Number of periods to look ahead for return calculation. 
                 Default is -24 (2 hours forward for 5-minute data).
        """
        self.dataframe["Return"] = self.dataframe["Close"].shift(lag) / self.dataframe["Close"] - 1
        
        self.dataframe["Signal"] = "WAIT"
        self.dataframe.loc[self.dataframe["Return"] > buy_threshold, "Signal"] = "BUY"
        self.dataframe.loc[self.dataframe["Return"] < sell_threshold, "Signal"] = "SELL"
  
    def define_features_target(self, features: list, target: str):
        """
        Separates the dataset into features/design matrix (X) and target (y).
        
        Args:
            features: List of column names to use as features.
            target: Name of the column to use as target variable.
        """
        self.X = self.dataframe[features]
        self.y = self.dataframe[target]
        
    def scale_data(self, fit: bool = True, scaler: StandardScaler = StandardScaler()):
        """
        Standardizes features using StandardScaler.
        
        Args:
            fit: Whether to fit a new scaler (True) or use existing one (False). Default is True.
            scaler: Scaler object to use. Default is StandardScaler().
        """
        if fit:
            scaled = scaler.fit_transform(self.X) 
        else:
            scaled = scaler.transform(self.X) 
        self.X = pd.DataFrame(scaled, columns=self.X.columns, index=self.X.index)
    

class Model:
    """
    A Support Vector Machine model for trading signal prediction.
    
    Attributes:
        svm: The SVM classifier (SVC by default).
        ypred: Predicted target values.
        f1: F1 score of the model.
    """
    
    def __init__(self, svm: sklearn.svm._classes.SVC = SVC(), kernel: str = "rbf", 
                 gamma: str = "scale", weight: str = "balanced", iter: int = 10_000):
        """
        Initializes the SVM model with specified parameters.
        
        Args:
            svm: SVM classifier instance. Default is SVC().
            kernel: Kernel type. Default is rbf/sigmoid.
            gamma: Kernel coefficient. Default is 'scale'.
            weight: Class weight handling. Default is 'balanced'.
            iter: Maximum number of iterations. Default is 10,000.
        """
        self.svm = svm
        self.svm.kernel = kernel
        self.svm.gamma = gamma
        self.svm.class_weight = weight
        self.svm.max_iter = iter
        self.ypred = None
        self.f1 = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the SVM model on the provided data.
        
        Args:
            X: Design matrix (features).
            y: Target values.
        """
        self.svm.fit(X, y)
        
    def predict(self, X: pd.DataFrame):
        """
        Generates predictions using the trained model.
        
        Args:
            X: Input features for prediction.
        """
        self.ypred = self.svm.predict(X)
        
    def f1_score(self, test: pd.Series, pred: pd.Series, average: str = "macro"):
        """
        Calculates the F1 score of the model's predictions.
        
        Args:
            test: True target values.
            pred: Predicted target values.
            average: F1 score averaging method. Default is 'macro'.
        """
        self.f1 = f1_score(test, pred, average=average)
        

class Backtest:
    """
    A backtesting engine for evaluating trading strategies.
    
    Attributes:
        capital: Initial investment capital. Default is $1,000,000.
        portfolio_value: List tracking portfolio value over time.
        active_long_pos: Dictionary with current long position details.
        active_short_pos: Dictionary with current short position details.
        n_shares: Number of shares traded per position. Default is 2,000.
        com: Commission rate per trade. Default is 0.125% (0.00125).
        stop_loss: Stop-loss threshold. Default is 10% (0.1).
        take_profit: Take-profit threshold. Default is 10% (0.1).
        calmar_ratio: Calculated Calmar ratio.
        sortino_ratio: Calculated Sortino ratio.
        sharpe_ratio: Calculated Sharpe ratio.
    """
    
    def __init__(self, capital: int = 1_000_000, n_shares: int = 2_000, 
                 com: float = 0.125 / 100, stop_loss: float = 0.1, 
                 take_profit: float = 0.1):
        self.capital = capital
        self.portfolio_value = [capital]
        self.active_long_pos = None
        self.active_short_pos = None
        self.n_shares = n_shares
        self.com = com
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.calmar_ratio = 0
        self.sortino_ratio = 0
        self.sharpe_ratio = 0
        
    def calculate_portfolio(self, dataset: pd.DataFrame, capital: int = 1_000_000):
        """
        Simulates trading based on signals and tracks portfolio value.
        
        Args:
            dataset: DataFrame containing price data and trading signals.
            capital: Initial capital for simulation. Default is $1,000,000.
        """
        self.capital = capital
        self.active_long_pos = None
        self.active_short_pos = None
        self.portfolio_value = [self.capital]
        
        for i, row in dataset.iterrows():
            
            # Close active long position
            if self.active_long_pos:
                if row.Close < self.active_long_pos['stop_loss'] or row.Close > self.active_long_pos['take_profit']:
                    pnl = row.Close * self.n_shares * (1 - self.com)
                    self.capital += pnl
                    self.active_long_pos = None
            
            # Close active short position
            if self.active_short_pos:
                if row.Close > self.active_short_pos['stop_loss'] or row.Close < self.active_short_pos['take_profit']:
                    cost = row.Close * self.n_shares * (1 + self.com)
                    pnl = self.active_short_pos['entry'] * self.n_shares - cost
                    self.capital += pnl
                    self.active_short_pos = None
                
            # Open long position
            if row["Signal"] == 'BUY' and self.active_short_pos is None and self.active_long_pos is None:
                cost = row.Close * self.n_shares * (1 + self.com)
                if self.capital > cost:
                    self.capital -= cost
                    self.active_long_pos = {
                        'datetime': row.name,
                        'opened_at': row.Close,
                        'take_profit': row.Close * (1 + self.take_profit),
                        'stop_loss': row.Close * (1 - self.stop_loss)
                    }

            # Open short position
            if row["Signal"] == 'SELL' and self.active_short_pos is None and self.active_long_pos is None:
                cost = row.Close * self.n_shares * (self.com)
                self.capital -= cost
                self.active_short_pos = {
                    'datetime': row.name,
                    'entry': row.Close,
                    'take_profit': row.Close * (1 - self.take_profit),
                    'stop_loss': row.Close * (1 + self.stop_loss)
                }

            # Calculate current position value
            position_value = 0
            if self.active_long_pos:
                position_value = row.Close * self.n_shares
            elif self.active_short_pos:
                position_value = self.active_short_pos['entry'] * self.n_shares - row.Close * self.n_shares

            # Update total portfolio value
            self.portfolio_value.append(self.capital + position_value)
    
    def calculate_calmar(self, bars_per_year = 19_656):
        """
        Calculates the Calmar ratio (CAGR / Max Drawdown).
        
        Args:
            bars_per_year: Number of trading periods in a year. 
                          Default is 19,656 (for 5-minute bars).
        """
        initial_val = self.portfolio_value[0]
        final_val = self.portfolio_value[-1]
        n_bars = len(self.portfolio_value)
        
        # CAGR calculation
        if initial_val <= 0 or final_val <= 0:
            return 0.0
        
        cagr = (final_val / initial_val) ** (1 / (n_bars/bars_per_year)) - 1
        
        # Max Drawdown calculation
        max_so_far = self.portfolio_value[0]
        max_drawdown = 0
        for pv in self.portfolio_value:
            if pv > max_so_far:
                max_so_far = pv
            drawdown = (max_so_far - pv) / max_so_far
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        if max_drawdown == 0:
            return cagr if cagr > 0 else 0.0
        
        self.calmar_ratio = cagr / max_drawdown
        
    def calculate_sharpe(self, bars_per_year = 19_656, rfr = 0.041):
        """
        Calculates the Sharpe ratio (excess return per unit of risk).
        
        Args:
            bars_per_year: Number of trading periods in a year. Default is 19,656 (for 5-minute bars).
            rfr: Risk-free rate. Default is 4.1% (0.041).
        """
        returns = pd.Series(self.portfolio_value).pct_change().dropna()
        excess_returns = returns - (rfr / bars_per_year)
        mean_excess_return = excess_returns.mean()
        std_return = returns.std()

        if std_return == 0:
            self.sharpe_ratio = 0.0
            return
        
        self.sharpe_ratio = (mean_excess_return / std_return) * np.sqrt(bars_per_year)
        
    def calculate_sortino(self, bars_per_year = 19_656, rfr = 0.041):
        """
        Calculates the Sortino ratio (excess return per unit of downside risk).
        
        Args:
            bars_per_year: Number of trading periods in a year. Default is 19,656 (for 5-minute bars).
            rfr: Risk-free rate. Default is 4.1% (0.041).
        """
        returns = pd.Series(self.portfolio_value).pct_change().dropna()
        excess_returns = returns - (rfr / bars_per_year)
        negative_returns = excess_returns[excess_returns < 0]
        mean_excess_return = excess_returns.mean()
        std_negative = negative_returns.std()

        if std_negative == 0:
            self.sortino_ratio = 0.0
            return    
        
        self.sortino_ratio = (mean_excess_return / std_negative) * np.sqrt(bars_per_year)