from jupiter_backtesting.strategy import Strategy
from jupiter_backtesting.stats import SimpleStats
from jupiter_backtesting.utils import Utils
from typing import Dict, Any, Optional
import vectorbt as vbt
import pandas as pd

class Analyzer:
    def __init__(
        self,
        strategy: Strategy,
        params: Dict[str, Any],
        full_data: pd.DataFrame,
        timeframe: str,
        price_col: str = "close",
        test_size: float = 0,  # 0 = no split
        init_cash: float = 1000,
        fees: float = 0.0002, 
        slippage: float = 0.000,
        trade_side: Optional[str] = 'longonly'
    ):
        self.ss = SimpleStats(price_col=price_col)
        self.util = Utils()
        self.strategy = strategy
        self.params = params
        self.timeframe = timeframe
        self.test_size = test_size
        self.init_cash = init_cash
        self.fees = fees
        self.slippage = slippage


        # Data preparation
        self.full_data = self.util.validate_data(full_data)
        
        if test_size > 0:
            self.train_df, self.test_df = self.util.time_based_split(self.full_data, test_size)
            self.train_df = self.strategy.preprocess_data(self.train_df.copy(), params)
        else:
            self.train_df = self.strategy.preprocess_data(self.full_data.copy(), params)
            self.test_df = None

        # Signal generation
        self.signals = self.strategy.generate_signals(self.train_df, **params)
        self._validate_signals()

        # Portfolio
        self.pf = vbt.Portfolio.from_signals(
            close=self.train_df[self.ss.price_col],
            entries=self.signals['entries'],
            exits=self.signals['exits'],
            short_entries=self.signals.get('short_entries'),
            short_exits=self.signals.get('short_exits'),
            freq=self.timeframe,
            init_cash=init_cash,
            fees=fees,
            slippage=slippage,
            direction=trade_side
        )

    def _validate_signals(self):
        if not self.signals['entries'].any():
            raise ValueError("No entry signals generated")
        if self.signals['entries'].index.difference(self.train_df.index).any():
            raise ValueError("Signal/data index mismatch")

    def oos_test(self) -> Optional[vbt.Portfolio]:
        """Out-of-sample test if test_size > 0"""
        if self.test_df is None or self.test_df.empty:
            return None

        test_df = self.strategy.preprocess_data(self.test_df.copy(), self.params)
        test_signals = self.strategy.generate_signals(test_df, **self.params)
        
        return vbt.Portfolio.from_signals(
            close=test_df[self.ss.price_col],
            entries=test_signals['entries'],
            exits=test_signals['exits'],
            freq=self.timeframe,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            direction='both'
        )
    
    def backtest_results(self) -> pd.DataFrame:
     """Returns a full performance summary of the backtest."""
     return self.ss.backtest_summary(self.pf, self.timeframe)

