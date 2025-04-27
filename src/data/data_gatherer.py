"""
Data gathering module for fetching market data from various sources
"""
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Union
import ccxt
from loguru import logger

class DataGatherer:
    """
    Handles data gathering from various sources including Yahoo Finance and crypto exchanges
    """
    
    def __init__(self, config: dict):
        """
        Initialize the data gatherer
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config
        self.setup_logging()
        self.setup_exchanges()
        
    def setup_logging(self):
        """Configure logging"""
        logger.add(
            "logs/data_gatherer_{time}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO"
        )
        
    def setup_exchanges(self):
        """Setup exchange connections"""
        try:
            # Initialize exchange connections if API keys are provided
            self.exchanges = {}
            if self.config.get('binance_api_key') and self.config.get('binance_secret'):
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': self.config['binance_api_key'],
                    'secret': self.config['binance_secret'],
                    'enableRateLimit': True
                })
                
            if self.config.get('ftx_api_key') and self.config.get('ftx_secret'):
                self.exchanges['ftx'] = ccxt.ftx({
                    'apiKey': self.config['ftx_api_key'],
                    'secret': self.config['ftx_secret'],
                    'enableRateLimit': True
                })
                
            logger.info(f"Initialized {len(self.exchanges)} exchange connections")
            
        except Exception as e:
            logger.error(f"Error setting up exchanges: {str(e)}")
            raise
            
    def get_stock_data(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance
        
        Args:
            symbols: Stock symbol(s) to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
                
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
                
                if df.empty:
                    logger.warning(f"No data found for {symbol}")
                    continue
                    
                # Standardize column names
                df.columns = [col.lower() for col in df.columns]
                df = df.rename(columns={
                    'stock splits': 'splits',
                    'dividends': 'dividend'
                })
                
                data[symbol] = df
                logger.info(f"Fetched {len(df)} rows for {symbol}")
                
            if len(data) == 1:
                return data[symbols[0]]
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise
            
    def get_crypto_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = '1d',
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch cryptocurrency data from specified exchange
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            exchange: Exchange name (must be initialized in setup_exchanges)
            timeframe: Data timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d', etc.)
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if exchange not in self.exchanges:
                raise ValueError(f"Exchange {exchange} not initialized")
                
            # Fetch OHLCV data
            ohlcv = self.exchanges[exchange].fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} rows for {symbol} from {exchange}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching crypto data: {str(e)}")
            raise
            
    def get_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: List[str]
    ) -> pd.DataFrame:
        """
        Calculate technical indicators for the given data
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicators to calculate
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            df = df.copy()
            
            for indicator in indicators:
                if indicator == 'sma':
                    # Simple Moving Average
                    df['sma_20'] = df['close'].rolling(window=20).mean()
                    df['sma_50'] = df['close'].rolling(window=50).mean()
                    df['sma_200'] = df['close'].rolling(window=200).mean()
                    
                elif indicator == 'ema':
                    # Exponential Moving Average
                    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
                    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
                    
                elif indicator == 'macd':
                    # MACD
                    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = ema_12 - ema_26
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                elif indicator == 'rsi':
                    # Relative Strength Index
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    
                elif indicator == 'bb':
                    # Bollinger Bands
                    sma = df['close'].rolling(window=20).mean()
                    std = df['close'].rolling(window=20).std()
                    df['bb_upper'] = sma + (std * 2)
                    df['bb_middle'] = sma
                    df['bb_lower'] = sma - (std * 2)
                    
            logger.info(f"Added technical indicators: {indicators}")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
            
    def get_sentiment_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch and analyze sentiment data for a given symbol
        
        Args:
            symbol: Stock/crypto symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with sentiment scores
        """
        try:
            # This is a placeholder for sentiment analysis
            # In a real implementation, you would:
            # 1. Fetch news articles/social media posts
            # 2. Process them through FinBERT or similar
            # 3. Aggregate sentiment scores
            logger.warning("Sentiment analysis not yet implemented")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching sentiment data: {str(e)}")
            raise
