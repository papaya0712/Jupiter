import pandas as pd
import numpy as np
import vectorbt as vbt

from hyperopt import Trials, tpe, hp, fmin, space_eval, STATUS_FAIL, STATUS_OK
from numpy.random import RandomState
from typing import Dict, Tuple
from quantybt.analyzer import Analyzer
from quantybt.stats import SimpleStats

class Optimizer:
    def __init__(
        self,
        analyzer,
        max_evals: int = 25,
        target_metric: str = "sharpe_ratio",
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
        self.ss = analyzer.ss

    def _objective(self, params):
        try:
            df = self.train_df.copy()
            df = self.strategy.preprocess_data(df, params)
            signals = self.strategy.generate_signals(df, **params)

            pf = vbt.Portfolio.from_signals(
                close=df[self.ss.price_col],
                entries=signals["entries"],
                exits=signals["exits"],
                freq=self.timeframe,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                direction="both",
            )

            metric_value = getattr(pf, self.target_metric)()
            return {"loss": -metric_value, "status": STATUS_OK}
        except Exception:
            return {"loss": np.inf, "status": STATUS_FAIL}

    def optimize(self):
        trials = Trials()
        rstate = RandomState(69)
        best = fmin(
            fn=self._objective,
            space=self.strategy.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=rstate,
        )
        return space_eval(self.strategy.param_space, best), trials