"""
Advanced Feature Engineering for Stock Market Prediction
Implements technical indicators, volatility measures, and market microstructure features
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Production-grade feature engineering for time series stock data
    Generates 40+ technical indicators with proper handling of edge cases
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.data = data.copy()
        self.required_cols = ['open', 'high', 'low', 'close', 'volume']
        self._validate_data()
        
    def _validate_data(self):
        """Ensure required columns exist"""
        missing = set(self.required_cols) - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def create_all_features(self) -> pd.DataFrame:
        """
        Generate comprehensive feature set:
        - Price-based indicators
        - Volume indicators
        - Volatility measures
        - Momentum indicators
        - Pattern recognition features
        """
        df = self.data.copy()
        
        # Price transformations
        df = self._add_price_features(df)
        
        # Moving averages
        df = self._add_moving_averages(df)
        
        # Momentum indicators
        df = self._add_momentum_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Volume indicators
        df = self._add_volume_indicators(df)
        
        # Pattern recognition
        df = self._add_pattern_features(df)
        
        # Market microstructure
        df = self._add_microstructure_features(df)
        
        # Target variable (classification: -1=down, 0=neutral, 1=up)
        df = self._create_target(df)
        
        # Remove NaN rows from indicator calculation
        df = df.dropna()
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic price transformations and returns"""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price range
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple and Exponential Moving Averages"""
        periods = [5, 10, 20, 50, 200]
        
        for period in periods:
            # Simple Moving Average
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            # Exponential Moving Average
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            # Price relative to MA
            df[f'close_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # Golden cross / Death cross signal
        df['ma_cross_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, MACD, Stochastic, ROC, Williams %R"""
        
        # RSI (Relative Strength Index)
        df = self._calculate_rsi(df, period=14)
        
        # MACD (Moving Average Convergence Divergence)
        df = self._calculate_macd(df)
        
        # Stochastic Oscillator
        df = self._calculate_stochastic(df, period=14)
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) 
                                   / df['close'].shift(period)) * 100
        
        # Williams %R
        df = self._calculate_williams_r(df, period=14)
        
        # Momentum
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD indicator with signal line and histogram"""
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # MACD cross signals
        df['macd_bullish_cross'] = (
            (df['macd'] > df['macd_signal']) & 
            (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        ).astype(int)
        
        return df
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Stochastic Oscillator"""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Williams %R"""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands, ATR, Historical Volatility"""
        
        # Bollinger Bands
        df = self._calculate_bollinger_bands(df, period=20, std_dev=2)
        
        # Average True Range
        df = self._calculate_atr(df, period=14)
        
        # Historical Volatility (20-day rolling std of returns)
        df['hist_volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Volatility ratio
        df['volatility_ratio'] = (
            df['returns'].rolling(window=5).std() / 
            df['returns'].rolling(window=20).std()
        )
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, 
                                   period: int = 20, 
                                   std_dev: int = 2) -> pd.DataFrame:
        """Bollinger Bands with %B indicator"""
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (bb_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std_dev)
        
        # %B indicator (position within bands)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Bandwidth
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=period).mean()
        
        # Normalized ATR
        df['atr_percent'] = df['atr'] / df['close']
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators"""
        
        # Volume moving averages
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume-Weighted Average Price (VWAP) - daily reset approximation
        df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        # Money Flow Index (MFI)
        df = self._calculate_mfi(df, period=14)
        
        return df
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi_ratio = positive_mf / negative_mf
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick patterns and price action features"""
        
        # Body size
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['body_size_pct'] = df['body_size'] / df['open']
        
        # Upper and lower shadows
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Doji detection (small body)
        df['is_doji'] = (df['body_size_pct'] < 0.001).astype(int)
        
        # Gap detection
        df['gap_up'] = (df['open'] > df['high'].shift(1)).astype(int)
        df['gap_down'] = (df['open'] < df['low'].shift(1)).astype(int)
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features"""
        
        # Bid-ask spread proxy (high-low spread)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Price efficiency (how much close deviates from VWAP)
        df['price_efficiency'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Volume concentration
        df['volume_concentration'] = df['volume'] / df['volume'].rolling(window=5).sum()
        
        return df
    
    def _create_target(self, df: pd.DataFrame, 
                      threshold: float = 0.001) -> pd.DataFrame:
        """
        Create target variable for classification:
        -1: Down movement (< -threshold)
         0: Neutral (-threshold <= x <= threshold)
         1: Up movement (> threshold)
        """
        future_returns = df['close'].shift(-1) / df['close'] - 1
        
        df['target'] = 0  # Neutral
        df.loc[future_returns < -threshold, 'target'] = -1  # Down
        df.loc[future_returns > threshold, 'target'] = 1   # Up
        
        # Drop last row (no future return)
        df = df[:-1]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of engineered feature names"""
        df = self.create_all_features()
        exclude = self.required_cols + ['target']
        return [col for col in df.columns if col not in exclude]


# Example usage
if __name__ == "__main__":
    # Sample data (replace with actual data loader)
    sample_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 1000)
    })
    
    # Create features
    engineer = FeatureEngineer(sample_data)
    features_df = engineer.create_all_features()
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Engineered shape: {features_df.shape}")
    print(f"\nFeature columns ({len(engineer.get_feature_names())}):")
    print(engineer.get_feature_names()[:10], "...")
    print(f"\nTarget distribution:")
    print(features_df['target'].value_counts())