"""
Risk management module for trading system
"""
import numpy as np
from typing import Dict, Tuple
from loguru import logger
from ..config.trading_config import TradingConfig

class RiskManager:
    """
    Handles risk management, position sizing, and portfolio protection
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize risk manager
        
        Args:
            config: Trading configuration object
        """
        self.config = config
        self.max_position_size = config.MAX_POSITION_SIZE
        self.risk_per_trade = config.RISK_PER_TRADE
        self.max_drawdown = config.MAX_DRAWDOWN
        
        # Track portfolio metrics
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.positions = {}
        
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        strategy: str = 'risk'
    ) -> float:
        """
        Calculate appropriate position size based on risk management rules
        
        Args:
            capital: Available trading capital
            entry_price: Planned entry price
            stop_loss: Stop loss price
            strategy: Position sizing strategy ('fixed', 'risk', 'kelly')
            
        Returns:
            Position size in base currency
        """
        try:
            if strategy == 'fixed':
                # Use fixed percentage of capital
                return capital * self.max_position_size
                
            elif strategy == 'risk':
                # Calculate position size based on risk per trade
                risk_amount = capital * self.risk_per_trade
                price_risk = abs(entry_price - stop_loss)
                position_size = risk_amount / price_risk
                
                # Ensure we don't exceed max position size
                max_size = capital * self.max_position_size
                return min(position_size, max_size)
                
            elif strategy == 'kelly':
                # Implement Kelly Criterion for position sizing
                # Note: This requires win rate and risk/reward ratio statistics
                win_rate = 0.5  # Should be calculated from historical data
                risk_reward = abs(entry_price - stop_loss)  # Should include take profit
                kelly_size = win_rate - ((1 - win_rate) / risk_reward)
                
                # Apply half-kelly for more conservative sizing
                position_size = (kelly_size * 0.5) * capital
                
                # Ensure we don't exceed max position size
                max_size = capital * self.max_position_size
                return min(position_size, max_size)
                
            else:
                raise ValueError(f"Unknown position sizing strategy: {strategy}")
                
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
            
    def calculate_stop_loss(
        self,
        entry_price: float,
        position_type: str,
        atr: float = None
    ) -> float:
        """
        Calculate stop loss price based on various methods
        
        Args:
            entry_price: Entry price of the position
            position_type: 'long' or 'short'
            atr: Average True Range value (optional)
            
        Returns:
            Stop loss price
        """
        try:
            if atr is not None:
                # Use ATR-based stop loss
                multiplier = 2.0  # Adjustable ATR multiplier
                if position_type == 'long':
                    return entry_price - (atr * multiplier)
                else:
                    return entry_price + (atr * multiplier)
            else:
                # Use fixed percentage stop loss
                if position_type == 'long':
                    return entry_price * (1 - self.config.STOP_LOSS_PCT)
                else:
                    return entry_price * (1 + self.config.STOP_LOSS_PCT)
                    
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return entry_price
            
    def calculate_take_profit(
        self,
        entry_price: float,
        position_type: str,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price of the position
            position_type: 'long' or 'short'
            risk_reward_ratio: Desired risk/reward ratio
            
        Returns:
            Take profit price
        """
        try:
            stop_loss = self.calculate_stop_loss(entry_price, position_type)
            risk = abs(entry_price - stop_loss)
            
            if position_type == 'long':
                return entry_price + (risk * risk_reward_ratio)
            else:
                return entry_price - (risk * risk_reward_ratio)
                
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return entry_price
            
    def update_portfolio_metrics(
        self,
        portfolio_value: float,
        timestamp: str
    ) -> Dict[str, float]:
        """
        Update portfolio metrics and check risk limits
        
        Args:
            portfolio_value: Current portfolio value
            timestamp: Current timestamp
            
        Returns:
            Dictionary with updated metrics
        """
        try:
            # Update peak value
            self.peak_value = max(self.peak_value, portfolio_value)
            
            # Calculate current drawdown
            self.current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            
            metrics = {
                'peak_value': self.peak_value,
                'current_drawdown': self.current_drawdown,
                'drawdown_limit': self.max_drawdown,
                'risk_level': self.current_drawdown / self.max_drawdown
            }
            
            # Log warning if approaching drawdown limit
            if self.current_drawdown > (self.max_drawdown * 0.8):
                logger.warning(
                    f"Approaching maximum drawdown limit. "
                    f"Current: {self.current_drawdown:.2%}, "
                    f"Max: {self.max_drawdown:.2%}"
                )
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {str(e)}")
            return {}
            
    def should_trade(self, portfolio_metrics: Dict[str, float]) -> Tuple[bool, str]:
        """
        Determine if trading should continue based on risk metrics
        
        Args:
            portfolio_metrics: Dictionary of current portfolio metrics
            
        Returns:
            Tuple of (should_trade, reason)
        """
        try:
            # Check drawdown limit
            if portfolio_metrics['current_drawdown'] >= self.max_drawdown:
                return False, "Maximum drawdown limit reached"
                
            # Check risk level
            if portfolio_metrics['risk_level'] >= 1.0:
                return False, "Risk level too high"
                
            return True, "Trading allowed"
            
        except Exception as e:
            logger.error(f"Error checking trading conditions: {str(e)}")
            return False, f"Error: {str(e)}"
