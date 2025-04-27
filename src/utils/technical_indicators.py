"""
Technical indicators calculation module
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger
import talib

class TechnicalIndicators:
    """
    Calculate technical indicators for trading analysis
    """
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Ensure we have required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError("DataFrame missing required OHLCV columns")
            
            # Create copy to avoid modifying original
            df = df.copy()
            
            # Add each indicator
            df = TechnicalIndicators.add_trend_indicators(df)
            df = TechnicalIndicators.add_momentum_indicators(df)
            df = TechnicalIndicators.add_volatility_indicators(df)
            df = TechnicalIndicators.add_volume_indicators(df)
            
            # Fill any NaN values with 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df

    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based technical indicators
        """
        try:
            # Simple Moving Averages
            df['sma_5'] = talib.SMA(df['close'], timeperiod=5)
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            
            # Exponential Moving Averages
            df['ema_5'] = talib.EMA(df['close'], timeperiod=5)
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'],
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Parabolic SAR
            df['sar'] = talib.SAR(df['high'], df['low'])
            
            # Average Directional Index
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
            return df

    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based technical indicators
        """
        try:
            # Relative Strength Index
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(
                df['high'],
                df['low'],
                df['close'],
                fastk_period=5,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=3,
                slowd_matype=0
            )
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            
            # Rate of Change
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            
            # Williams %R
            df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Money Flow Index
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
            return df

    @staticmethod
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based technical indicators
        """
        try:
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                df['close'],
                timeperiod=20,
                nbdevup=2,
                nbdevdn=2,
                matype=0
            )
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            # Average True Range
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Standard Deviation
            df['stddev'] = talib.STDDEV(df['close'], timeperiod=20)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {str(e)}")
            return df

    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based technical indicators
        """
        try:
            # On-Balance Volume
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
            # Chaikin A/D Line
            df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            
            # Volume Rate of Change
            df['vroc'] = talib.ROC(df['volume'], timeperiod=1)
            
            # Price Volume Trend
            df['pvt'] = (df['close'].pct_change() * df['volume']).cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return df

    @staticmethod
    def get_feature_names() -> List[str]:
        """
        Get list of all technical indicator feature names
        """
        return [
            # Trend Indicators
            'sma_5', 'sma_20', 'sma_50',
            'ema_5', 'ema_20', 'ema_50',
            'macd', 'macd_signal', 'macd_hist',
            'sar', 'adx',
            
            # Momentum Indicators
            'rsi', 'stoch_k', 'stoch_d',
            'roc', 'willr', 'mfi',
            
            # Volatility Indicators
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'stddev',
            
            # Volume Indicators
            'obv', 'ad', 'vroc', 'pvt'
        ]
