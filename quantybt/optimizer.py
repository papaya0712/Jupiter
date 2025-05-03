import pandas as pd
import numpy as np
import vectorbt as vbt
import plotly.graph_objects as go
import logging  

from dataclasses import dataclass
from typing import Dict, Any, Sequence, Tuple, List, Optional
from hyperopt import space_eval, STATUS_OK, tpe, fmin, Trials
from quantybt.plots import _PlotTrainTestSplit, _PlotGeneralization
from quantybt.analyzer import Analyzer
from quantybt.stats import Stats

logger = logging.getLogger(__name__)  

class SimpleOptimizer:
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
    
# ==================================== advanced Optimizer ==================================== # 

def _to_dict(obj: Any) -> Dict:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0].to_dict()
        return {col: obj[col].to_dict() for col in obj.columns}
    return dict(obj)

@dataclass
class _WFOSplitCfg:
    n_folds: int = 3
    test_size: float = 0.3
    min_train_size: float = 0.2

class AdvancedOptimizer:
    def __init__(
        self,
        analyzer,
        max_evals: int = 25,
        target_metric: str = "sharpe_ratio",
        beta: float = 0.3,
        split_cfg: _WFOSplitCfg | Sequence[_WFOSplitCfg] = _WFOSplitCfg(),
    ):
        if analyzer.test_size <= 0:
            raise ValueError("Analyzer must use test_size > 0 for optimization")
        self.analyzer = analyzer
        self.strategy = analyzer.strategy
        self.timeframe = analyzer.timeframe
        self.max_evals = max_evals
        self.target_metric = target_metric
        self.beta = beta
        self.init_cash = analyzer.init_cash
        self.fees = analyzer.fees
        self.slippage = analyzer.slippage
        self.s = analyzer.s
        self.split_cfgs: List[_WFOSplitCfg] = (
            [split_cfg] if isinstance(split_cfg, _WFOSplitCfg)
            else list(split_cfg)
        )
        self._splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = self._prepare_splits()
        self.best_params: Optional[Dict[str, Any]] = None
        self.trials: Optional[Trials] = None
        self.train_pf: Optional[vbt.Portfolio] = None
        self.test_pf: Optional[vbt.Portfolio] = None
        self._history_diffs: List[float] = []
        self._history_gl_max: List[float] = []
        self.trial_metrics: List[Tuple[float, float]] = []
        self.metrics_map = {
            "sharpe_ratio": lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[0],
            "sortino_ratio": lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[1],
            "calmar_ratio": lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[2],
            "total_return": lambda pf: self.s._returns(pf)[0],
            "max_drawdown": lambda pf: self.s._risk_metrics(self.timeframe, pf)[0],
            "volatility": lambda pf: self.s._risk_metrics(self.timeframe, pf)[2],
            "profit_factor": lambda pf: pf.stats().get("Profit Factor", np.nan),
        }

    def _generate_anchored_splits(self, df: pd.DataFrame, cfg: _WFOSplitCfg) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        total_samples = len(df)
        test_samples = int(cfg.test_size * total_samples)
        min_train_samples = int(cfg.min_train_size * total_samples)
        if min_train_samples + cfg.n_folds * test_samples > total_samples:
            raise ValueError(
                f"Not enough data for {cfg.n_folds} folds"
            )
        splits = []
        current_start = min_train_samples
        for _ in range(cfg.n_folds):
            if current_start + test_samples > total_samples:
                break
            train_df = df.iloc[:current_start]
            val_df = df.iloc[current_start:current_start + test_samples]
            splits.append((train_df, val_df))
            current_start += test_samples
        return splits

    def _prepare_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        df = self.analyzer.train_df.sort_index()
        all_splits = []
        for cfg in self.split_cfgs:
            all_splits.extend(self._generate_anchored_splits(df, cfg))
        return all_splits

    def _metric(self, pf: vbt.Portfolio) -> float:
        if self.target_metric in self.metrics_map:
            return self.metrics_map[self.target_metric](pf)
        try:
            return getattr(pf, self.target_metric)()
        except Exception:
            return pf.stats().get(self.target_metric, np.nan)

    @staticmethod
    def _choose_direction(sig: Dict[str, Any]) -> str:
        has_short = sig.get("short_entries") is not None or sig.get("short_exits") is not None
        return "all" if has_short else "longonly"

    def _objective(self, params: dict) -> dict:
     try:
        
        seed = int(abs(hash(frozenset(params.items()))) % 2**32)
        np.random.seed(seed)

        losses, is_metrics, val_metrics = [], [], []
        higher_is_better = self.target_metric not in ["max_drawdown", "volatility"]

        for train_df, val_df in self._splits:
            df_train = self.strategy.preprocess_data(train_df.copy(), params)
            sig_train = self.strategy.generate_signals(df_train, **params)
            pf_train = vbt.Portfolio.from_signals(
                close=df_train[self.s.price_col],
                entries=sig_train.get("entries"),
                exits=sig_train.get("exits"),
                short_entries=sig_train.get("short_entries"),
                short_exits=sig_train.get("short_exits"),
                freq=self.timeframe,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                direction=self._choose_direction(sig_train),
                sl_stop=params.get("sl_pct"),
                tp_stop=params.get("tp_pct"),
            )
            m_is = self._metric(pf_train)

            df_val = self.strategy.preprocess_data(val_df.copy(), params)
            sig_val = self.strategy.generate_signals(df_val, **params)
            pf_val = vbt.Portfolio.from_signals(
                close=df_val[self.s.price_col],
                entries=sig_val.get("entries"),
                exits=sig_val.get("exits"),
                short_entries=sig_val.get("short_entries"),
                short_exits=sig_val.get("short_exits"),
                freq=self.timeframe,
                init_cash=self.init_cash,
                fees=self.fees,
                slippage=self.slippage,
                direction=self._choose_direction(sig_val),
                sl_stop=params.get("sl_pct"),
                tp_stop=params.get("tp_pct"),
            )
            m_val = self._metric(pf_val)

            # Compute generalization loss with clipping to [0,1]
            if higher_is_better:
                if m_is <= 0 or not np.isfinite(m_is) or not np.isfinite(m_val):
                    gl = 1.0
                else:
                    raw_gl = 1.0 - (m_val / m_is)
                    gl = max(0.0, min(1.0, raw_gl))
            else:
                if m_val <= 0 or not np.isfinite(m_is) or not np.isfinite(m_val):
                    gl = 1.0
                else:
                    raw_gl = 1.0 - (m_is / m_val)
                    gl = max(0.0, min(1.0, raw_gl))

            # Collect losses and metrics
            losses.append((-m_val, gl))
            is_metrics.append(m_is)
            val_metrics.append(m_val)

        m_val_avg = -np.mean([l[0] for l in losses]) 
        gl_max    = max([l[1] for l in losses])          

        scale_raw = np.std(self._history_diffs[-10:]) if len(self._history_diffs) >= 10 else 1.0
        scale     = np.clip(scale_raw if scale_raw > 0 else 1.0, 0.1, 10.0)

        loss = -m_val_avg + self.beta * (gl_max / scale)

        self._history_gl_max.append(gl_max)
        self._history_diffs.append(loss)
        self.trial_metrics.append((np.mean(is_metrics), np.mean(val_metrics)))

        return {"loss": loss, "status": STATUS_OK, "params": params}

     except Exception as e:
        logger.error(f"Objective error: {e}", exc_info=True)
        return {"loss": np.inf, "status": STATUS_OK}

    def optimize(self) -> Tuple[dict, Trials]:
        self.trials = Trials()
        best = fmin(
            fn=self._objective,
            space=self.strategy.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=self.trials,
            rstate=np.random.default_rng(42),
        )
        self.best_params = space_eval(self.strategy.param_space, best)
        return self.best_params, self.trials

    def evaluate(self) -> Dict[str, Any]:
        if self.best_params is None:
            raise ValueError("Call .optimize() first")
        df_train_full = self.strategy.preprocess_data(self.analyzer.train_df.copy(), self.best_params)
        sig_train = self.strategy.generate_signals(df_train_full, **self.best_params)
        self.train_pf = vbt.Portfolio.from_signals(
            close=df_train_full[self.s.price_col],
            entries=sig_train.get("entries"),
            exits=sig_train.get("exits"),
            short_entries=sig_train.get("short_entries"),
            short_exits=sig_train.get("short_exits"),
            freq=self.timeframe,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            direction=self._choose_direction(sig_train),
            sl_stop=self.best_params.get("sl_pct"),
            tp_stop=self.best_params.get("tp_pct"),
        )
        df_test = self.strategy.preprocess_data(self.analyzer.test_df.copy(), self.best_params)
        sig_test = self.strategy.generate_signals(df_test, **self.best_params)
        self.test_pf = vbt.Portfolio.from_signals(
            close=df_test[self.s.price_col],
            entries=sig_test.get("entries"),
            exits=sig_test.get("exits"),
            short_entries=sig_test.get("short_entries"),
            short_exits=sig_test.get("short_exits"),
            freq=self.timeframe,
            init_cash=self.init_cash,
            fees=self.fees,
            slippage=self.slippage,
            direction=self._choose_direction(sig_test),
            sl_stop=self.best_params.get("sl_pct"),
            tp_stop=self.best_params.get("tp_pct"),
        )
        train_summary = _to_dict(self.train_pf.stats())
        test_summary = _to_dict(self.test_pf.stats())
        return {
            "train_pf": self.train_pf,
            "test_pf": self.test_pf,
            "train_summary": train_summary,
            "test_summary": test_summary,
            "history_generalization_loss": self._history_diffs,
        }

    def plot_train_test(self, title: str = "Train vs Hold-out Performance", export_html: bool = False, export_image: bool = False, file_name: str = "train_test_plot"):
        return _PlotTrainTestSplit(self).plot_oos(title=title, export_html=export_html, export_image=export_image, file_name=file_name)

    def plot_generalization(self, title: str = "Generalization Performance"):
        return _PlotGeneralization(self).plot_generalization(title=title)