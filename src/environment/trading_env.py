"""
Custom OpenAI Gym environment for trading with LSTM-PPO
"""
import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple
from ..config.trading_config import TradingConfig
from ..utils.technical_indicators import calculate_indicators

class TradingEnvironment(gym.Env):
    """
    Trading environment that implements OpenAI gym interface
    """
    
    def __init__(self, data: pd.DataFrame, config: TradingConfig):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.config = config
        self.current_step = 0
        self.initial_balance = config.INITIAL_BALANCE
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        
        # Calculate technical indicators
        self.data = calculate_indicators(self.data, config.TECHNICAL_INDICATORS)
        
        # Define action space (buy, sell, hold)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        n_features = len(self.data.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(config.WINDOW_SIZE, n_features),
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state
        """
        self.current_step = self.config.WINDOW_SIZE
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        """
        # Get current price data
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute trading action
        reward = 0
        done = False
        info = {}
        
        if action == 0:  # Buy
            if self.position <= 0:
                max_shares = (self.balance * self.config.MAX_POSITION_SIZE) // current_price
                self.position = max_shares
                self.balance -= max_shares * current_price * (1 + self.config.TRADING_FEES)
                self.trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'shares': max_shares,
                    'step': self.current_step
                })
                
        elif action == 1:  # Sell
            if self.position >= 0:
                sell_value = self.position * current_price * (1 - self.config.TRADING_FEES)
                self.balance += sell_value
                self.position = 0
                self.trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'shares': self.position,
                    'step': self.current_step
                })
        
        # Calculate reward
        portfolio_value = self.balance + (self.position * current_price)
        reward = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Apply drawdown penalty
        max_portfolio_value = max(trade['price'] * trade['shares'] + self.initial_balance 
                                for trade in self.trades) if self.trades else self.initial_balance
        drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
        if drawdown > self.config.MAX_DRAWDOWN:
            reward -= drawdown
        
        # Update state
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Prepare info dict
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'trades': len(self.trades),
            'drawdown': drawdown
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current state observation
        """
        # Get window of data
        start = self.current_step - self.config.WINDOW_SIZE
        end = self.current_step
        window_data = self.data.iloc[start:end].values
        
        # Normalize data
        window_mean = window_data.mean(axis=0)
        window_std = window_data.std(axis=0)
        normalized_data = (window_data - window_mean) / (window_std + 1e-8)
        
        return normalized_data.astype(np.float32)
    
    def render(self, mode='human'):
        """
        Render the environment
        """
        portfolio_value = self.balance + (self.position * self.data.iloc[self.current_step]['close'])
        print(f'Step: {self.current_step}')
        print(f'Portfolio Value: {portfolio_value:.2f}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Position: {self.position}')
        print(f'Number of Trades: {len(self.trades)}')
