import pandas as pd
import numpy as np
import vectorbt as vbt

from hyperopt import Trials, tpe, fmin, space_eval, STATUS_FAIL, STATUS_OK
from typing import Dict, Tuple
from analyzer import Analyzer
from .stats import SimpleStats

class Optimizer: 
    def __init__(
        self,
        analyzer: Analyzer,
        max_evals: int = 25,
        target_metric: str = "sharpe_ratio"
    ):
        if analyzer.test_size <= 0:
            raise ValueError("Analyzer must use test_size > 0 for optimization")
        
        self.train_df = analyzer.train_df
        self.strategy = analyzer.strategy
        self.timeframe = analyzer.timeframe
        self.max_evals = max_evals
        self.target_metric = target_metric
        self.init_cash = analyzer.pf.init_cash
        self.fees = analyzer.pf.fees
        self.slippage = analyzer.pf.slippage
        self.ss = SimpleStats(price_col=analyzer.ss.price_col)

    def _objective(self, params: Dict) -> Dict:
        try:
            df = self.train_df.copy()
            df = self.strategy.preprocess_data(df, params)
            signals = self.strategy.generate_signals(df, **params)

            pf = vbt.Portfolio.from_signals(
                close=df[self.ss.price_col],
                entries=signals['entries'],
                exits=signals['exits'],
                freq=self.timeframe,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                direction='both'
            )
            
            metric_value = getattr(pf, self.target_metric)()
            return {'loss': -metric_value, 'status': STATUS_OK}
        except Exception as e:
            return {'loss': np.nan, 'status': STATUS_FAIL, 'error': str(e)}

    def optimize(self) -> Tuple[Dict, Trials]:
        trials = Trials()
        best = fmin(
            fn=self._objective,
            space=self.strategy.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials
        )
        return space_eval(self.strategy.param_space, best), trials
