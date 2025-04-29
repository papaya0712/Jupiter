import pandas as pd
import numpy as np
import vectorbt as vbt
import plotly.graph_objects as go
import logging  

from hyperopt import space_eval, STATUS_OK
from quantybt.plots import _PlotTrainTestSplit, _PlotGeneralization
from quantybt.analyzer import Analyzer
from quantybt.stats import Stats

logger = logging.getLogger(__name__)  

class TrainTestOptimizer:
    """
    Simple method to reduce overfitting: split your data into In-Sample (for training) 
    and Out-of-Sample (for testing on unseen data).

    Plot Functions:
     - plot_train_test() shows IS vs OOS equity curve & drawdowncurves and a summary table
     - plot_generalization() shows IS vs OOS perfromance for every trial in a scatterplot with linear Regression as generalization metrics
    """
    def __init__(
        self,
        analyzer,
        max_evals: int = 25,
        target_metric: str = "sharpe_ratio",):

        if analyzer.test_size <= 0:
            raise ValueError("Analyzer must use test_size > 0 for optimization")

        self.analyzer = analyzer
        self.strategy = analyzer.strategy
        self.timeframe = analyzer.timeframe
        self.max_evals = max_evals
        self.target_metric = target_metric
        self.init_cash = analyzer.init_cash
        self.fees = analyzer.fees
        self.slippage = analyzer.slippage
        self.s = analyzer.s

        self.best_params = None
        self.trials = None
        self.train_pf = None
        self.test_pf = None

        
        self.trial_metrics = []  

        # Metrics map
        self.metrics_map = {
            "sharpe_ratio": lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[0],
            "sortino_ratio": lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[1],
            "calmar_ratio": lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[2],
            "total_return": lambda pf: self.s._returns(pf)[0],
            "max_drawdown": lambda pf: self.s._risk_metrics(self.timeframe, pf)[0],
            "volatility": lambda pf: self.s._risk_metrics(self.timeframe, pf)[2],
            "profit_factor": lambda pf: pf.stats().get("Profit Factor", np.nan),
        }

    def _get_metric_value(self, pf: vbt.Portfolio) -> float:
        if self.target_metric in self.metrics_map:
            return self.metrics_map[self.target_metric](pf)
        try:
            return getattr(pf, self.target_metric)()
        except Exception:
            return pf.stats().get(self.target_metric, np.nan)

    def _objective(self, params: dict) -> dict:
        try:
            seed = int(abs(hash(frozenset(params.items())))) % 2**32  
            np.random.seed(seed)
            # In-Sample
            df_is = self.analyzer.train_df.copy()
            df_is = self.strategy.preprocess_data(df_is, params)
            sig_is = self.strategy.generate_signals(df_is, **params)
            pf_is = vbt.Portfolio.from_signals(
                close=df_is[self.s.price_col],
                entries=sig_is.get('entries'), exits=sig_is.get('exits'),
                short_entries=sig_is.get('short_entries'), short_exits=sig_is.get('short_exits'),
                freq=self.timeframe, init_cash=self.init_cash,
                fees=self.fees, slippage=self.slippage,
                direction='longonly', sl_stop=params.get('sl_pct'), tp_stop=params.get('tp_pct')
            )
            val_is = self._get_metric_value(pf_is)

            # Out-of-Sample
            df_oos = self.analyzer.test_df.copy()
            df_oos = self.strategy.preprocess_data(df_oos, params)
            sig_oos = self.strategy.generate_signals(df_oos, **params)
            pf_oos = vbt.Portfolio.from_signals(
                close=df_oos[self.s.price_col],
                entries=sig_oos.get('entries'), exits=sig_oos.get('exits'),
                short_entries=sig_oos.get('short_entries'), short_exits=sig_oos.get('short_exits'),
                freq=self.timeframe, init_cash=self.init_cash,
                fees=self.fees, slippage=self.slippage,
                direction='longonly', sl_stop=params.get('sl_pct'), tp_stop=params.get('tp_pct')
            )
            val_oos = self._get_metric_value(pf_oos)

            # loss function 
            penalty = 0.5 * abs(val_is - val_oos) / (np.std(self.trial_metrics) + 1e-6)  
            loss = -val_is + penalty  
            
            self.trial_metrics.append((val_is, val_oos))  
            return {"loss": loss, "status": STATUS_OK, "params": params} 
        
        except Exception as e:  
          logger.error(f"Error with params {params}: {e}", exc_info=True)  
          return {"loss": np.inf, "status": STATUS_OK}  

    def optimize(self) -> tuple:
        from hyperopt import fmin, tpe, Trials
        trials = Trials()
        self.trials = trials
        best = fmin(
            fn=self._objective,
            space=self.strategy.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(42)
        )
        self.best_params = space_eval(self.strategy.param_space, best)
        return self.best_params, trials

    def evaluate(self) -> dict:
        if self.best_params is None:
            raise ValueError("Call optimize() before evaluate().")

        # Final In-Sample
        df_is = self.analyzer.train_df.copy()
        df_is = self.strategy.preprocess_data(df_is, self.best_params)
        sig_is = self.strategy.generate_signals(df_is, **self.best_params)
        self.train_pf = vbt.Portfolio.from_signals(
            close=df_is[self.s.price_col],
            entries=sig_is.get('entries'), exits=sig_is.get('exits'),
            short_entries=sig_is.get('short_entries'), short_exits=sig_is.get('short_exits'),
            freq=self.timeframe, init_cash=self.init_cash,
            fees=self.fees, slippage=self.slippage,
            direction='longonly', sl_stop=self.best_params.get('sl_pct'), tp_stop=self.best_params.get('tp_pct')
        )

        # Final Out-of-Sample
        df_oos = self.analyzer.test_df.copy()
        df_oos = self.strategy.preprocess_data(df_oos, self.best_params)
        sig_oos = self.strategy.generate_signals(df_oos, **self.best_params)
        self.test_pf = vbt.Portfolio.from_signals(
            close=df_oos[self.s.price_col],
            entries=sig_oos.get('entries'), 
            exits=sig_oos.get('exits'),
            short_entries=sig_oos.get('short_entries'), 
            short_exits=sig_oos.get('short_exits'),
            freq=self.timeframe, 
            init_cash=self.init_cash,
            fees=self.fees, 
            slippage=self.slippage,
            direction='longonly', 
            sl_stop=self.best_params.get('sl_pct'), 
            tp_stop=self.best_params.get('tp_pct')
        )

        # Summaries
        train_summary = self.s.backtest_summary(self.train_pf, self.timeframe)
        test_summary = self.s.backtest_summary(self.test_pf, self.timeframe)

        return {
            'train_pf': self.train_pf,
            'test_pf': self.test_pf,
            'train_summary': train_summary,
            'test_summary': test_summary,
            'trial_metrics': self.trial_metrics
        }
    
    def plot_train_test(self,
             title: str = 'In-Sample vs Out-of-Sample Performance',
             export_html: bool = False,
             export_image: bool = False,
             file_name: str = 'train_test_plot[QuantyBT]') -> go.Figure:
        
        plotter = _PlotTrainTestSplit(self)
        return plotter.plot_oos(title=title, export_html=export_html, export_image=export_image, file_name=file_name)
    
    def plot_generalization(self, title: str = "IS vs OOS Performance"):
     plotter = _PlotGeneralization(self)
     return plotter.plot_generalization(title=title)