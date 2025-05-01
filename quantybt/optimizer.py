import pandas as pd
import numpy as np
import vectorbt as vbt
import plotly.graph_objects as go
import logging  

from dataclasses import dataclass
from typing import Dict, Any, Sequence, Tuple, List
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
class _SplitCfg:
    frac_val: float = 0.3       
    random_state: int = 42         

class AdvancedOptimizer:
    """
    Advanced strategy optimizer that balances performance and robustness using Generalization Loss (GL).
    Designed to find parameters that perform well on unseen data while minimizing overfitting.

    Core intention: Penalizes parameter sets that show large performance gaps between training and validation splits.

    Key Features:
    - Multi-Split Cross-Validation: Tests parameters on multiple temporal splits to simulate different market regimes
    - Adaptive Regularization: Uses beta-controlled penalty term to enforce robustness

    Parameters:
    analyzer (Analyzer): Configured strategy analyzer with data and settings
    max_evals (int): Maximum optimization trials (100-500 recommended)
    target_metric (str): Optimization target ('sharpe_ratio', 'sortino_ratio', 'calmar_ratio', etc.)
    beta (float): Regularization strength [0-1]:
        - 0.3 : Aggressive optimization (prioritize performance)
        - 0.5 : Balanced approach (default)
        - 1.0 : Conservative optimization (maximize robustness)
    split_cfg (SplitCfg): Configuration for temporal validation splits
    
    Usage:
    >>> optimizer = TrainTestOptimizer(analyzer, max_evals=100, target_metric="sharpe_ratio", beta=0.7)
    >>> best_params, trials = optimizer.optimize()
    >>> results = optimizer.evaluate()
    """
    def __init__(
        self,
        analyzer,
        max_evals: int = 100,
        target_metric: str = "sharpe_ratio",
        beta: float = 0.5,
        split_cfg: _SplitCfg | Sequence[_SplitCfg] = _SplitCfg(),
    ):
        # --- Plausibilität -------------------------------------------------- #
        if analyzer.test_size <= 0:
            raise ValueError("Analyzer must use test_size > 0 for optimization")

        # --- Grundlegende Eigenschaften ------------------------------------ #
        self.analyzer = analyzer
        self.strategy = analyzer.strategy
        self.timeframe = analyzer.timeframe
        self.max_evals = max_evals
        self.target_metric = target_metric
        self.init_cash = analyzer.init_cash
        self.fees = analyzer.fees
        self.slippage = analyzer.slippage
        self.s = analyzer.s
        self.beta = beta

        # --- Split-Konfiguration ------------------------------------------- #
        self.split_cfgs: List[_SplitCfg] = (
            [split_cfg] if isinstance(split_cfg, _SplitCfg) else list(split_cfg)
        )
        self._splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = self._prepare_splits()

        # --- Platzhalter ---------------------------------------------------- #
        self.best_params: Dict[str, Any] | None = None
        self.trials: Trials | None = None
        self.train_pf: vbt.Portfolio | None = None
        self.test_pf: vbt.Portfolio | None = None

        # --- Historien für Analyse & Plotter -------------------------------- #
        self._history_diffs: List[float] = []              # Generalization-Loss pro Trial
        self.trial_metrics: List[Tuple[float, float]] = [] # (m_is̅, m_val̅)  – Komp. für alten Plotter

        # --- Metrik-Mapping ------------------------------------------------- #
        self.metrics_map = {
            "sharpe_ratio":   lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[0],
            "sortino_ratio":  lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[1],
            "calmar_ratio":   lambda pf: self.s._risk_adjusted_metrics(self.timeframe, pf)[2],
            "total_return":   lambda pf: self.s._returns(pf)[0],
            "max_drawdown":   lambda pf: self.s._risk_metrics(self.timeframe, pf)[0],
            "volatility":     lambda pf: self.s._risk_metrics(self.timeframe, pf)[2],
            "profit_factor":  lambda pf: pf.stats().get("Profit Factor", np.nan),
        }

    def _prepare_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """timebased Train/Val-Splits """
        df = self.analyzer.train_df.sort_index()
        splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        for cfg in self.split_cfgs:
            split_idx = int(len(df) * (1 - cfg.frac_val))
            train_df = df.iloc[:split_idx].copy()
            val_df   = df.iloc[split_idx:].copy()
            splits.append((train_df, val_df))
        return splits

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
       
        seed = int(abs(hash(frozenset(params.items())))) % 2**32
        np.random.seed(seed)

        losses, is_metrics, val_metrics = [], [], []
        metric_higher_is_better = self.target_metric not in ["max_drawdown", "volatility"]

        def compute_gl(m_is: float, m_val: float, higher_is_better: bool) -> float:
            try:
                if not np.isfinite(m_is) or not np.isfinite(m_val):
                    return 1.0

                if higher_is_better:
                    if m_is <= 0:
                        return 1.0
                    ratio = m_val / m_is
                else:
                    if m_val <= 0:
                        return 1.0
                    ratio = m_is / m_val

                return max(0.0, min(1.0, 1.0 - ratio))

            except Exception:
                return 1.0

        for train_df, val_df in self._splits:
            # --- Train ------------------------------------------ #
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

            # --- Validation ------------------------------------- #
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

            # --- Generalization Loss ---------------------------- #
            gl = compute_gl(m_is, m_val, metric_higher_is_better)

            is_metrics.append(m_is)
            val_metrics.append(m_val)
            losses.append((-m_val, gl))

            # Debugging Split
            print(f"[GL] Split: m_is={m_is:.4f}, m_val={m_val:.4f}, GL={gl:.4f}")

        m_val_bar = float(np.mean([l[0] for l in losses])) * -1
        gl_bar = float(np.mean([l[1] for l in losses]))


        self._history_diffs.append(gl_bar)
        scale_raw = np.std(self._history_diffs)
        scale = np.clip(scale_raw if scale_raw > 0 else 1.0, 0.1, 10.0)

        loss = -m_val_bar + self.beta * (gl_bar / scale)

        self.trial_metrics.append((float(np.mean(is_metrics)), float(np.mean(val_metrics))))

        print(f"[Trial Summary] m_val̅={m_val_bar:.4f}, GL̅={gl_bar:.4f}, Loss={loss:.4f}")

        return {"loss": loss, "status": STATUS_OK, "params": params}

     except Exception as e:
        logger.error(f"Objective-Fehler bei Params {params}: {e}", exc_info=True)
        return {"loss": np.inf, "status": STATUS_OK}
     
    def optimize(self) -> Tuple[dict, Trials]:
        """hyperopt start"""
        trials = Trials()
        best_space = fmin(
            fn=self._objective,
            space=self.strategy.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(42),
        )
        self.trials = trials
        self.best_params = space_eval(self.strategy.param_space, best_space)
        return self.best_params, trials

    def evaluate(self) -> Dict[str, Any]:
        """returns the train and test vbt stats summary"""
        if self.best_params is None:
            raise ValueError("Rufe zuerst .optimize() auf!")

       
        df_train_full = self.strategy.preprocess_data(
            self.analyzer.train_df.copy(), self.best_params
        )
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

        df_test = self.strategy.preprocess_data(
            self.analyzer.test_df.copy(), self.best_params
        )
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

        train_summary_raw = self.s.backtest_summary(self.train_pf, self.timeframe)
        test_summary_raw  = self.s.backtest_summary(self.test_pf,  self.timeframe)

        train_summary = _to_dict(train_summary_raw)
        test_summary  = _to_dict(test_summary_raw)

        test_stats = self.test_pf.stats()
        test_summary["max_drawdown"]  = test_stats.get("Max Drawdown")
        test_summary["profit_factor"] = test_stats.get("Profit Factor")

        return {
            "train_pf": self.train_pf,
            "test_pf":  self.test_pf,
            "train_summary": train_summary,
            "test_summary":  test_summary,
            "history_generalization_loss": self._history_diffs,
        }

    def plot_train_test(
        self,
        title: str = "Train vs Hold-out Performance",
        export_html: bool = False,
        export_image: bool = False,
        file_name: str = "train_test_plot[QuantyBT]",
    ):
        
        return _PlotTrainTestSplit(self).plot_oos(
            title=title,
            export_html=export_html,
            export_image=export_image,
            file_name=file_name,
        )

    def plot_generalization(self, title: str = "Generalization Performance"):
        return _PlotGeneralization(self).plot_generalization(title=title)

# ==================================== Walk-forward Optimizer (soon) ==================================== #

