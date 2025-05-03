import pandas as pd
import numpy as np
import holoviews as hv
import vectorbt as vbt
from typing import Dict, Callable, List, Any, Optional, Union, Sequence
from .plots import _PlotFinDiff
from .stats import Stats
from .analyzer import Analyzer

MetricKey = str
MetricFunc = Callable[["Stats", str, "vbt.Portfolio"], float]

class LocalSensitivityAnalyzer:
    _METRIC_MAPS: Dict[MetricKey, MetricFunc] = {
        "sharpe_ratio":  lambda s, tf, pf: s._risk_adjusted_metrics(tf, pf)[0],
        "sortino_ratio": lambda s, tf, pf: s._risk_adjusted_metrics(tf, pf)[1],
        "calmar_ratio":  lambda s, tf, pf: s._risk_adjusted_metrics(tf, pf)[2],
        "total_return":  lambda s, tf, pf: s._returns(pf)[0],
        "max_drawdown":  lambda s, tf, pf: s._risk_metrics(tf, pf)[0],
        "volatility":    lambda s, tf, pf: s._risk_metrics(tf, pf)[2],
        "profit_factor": lambda s, tf, pf: pf.stats().get("Profit Factor", np.nan),
    }

    def __init__(
        self,
        analyzer: "Analyzer",
        target_metrics: Union[MetricKey, Sequence[MetricKey]] = "sharpe_ratio",
        seed: Optional[int] = 123,
    ):
        if isinstance(target_metrics, str):
            self.metrics: List[MetricKey] = [target_metrics]
        else:
            self.metrics = list(target_metrics)

        unknown = [m for m in self.metrics if m not in self._METRIC_MAPS]
        if unknown:
            raise ValueError(f"unknown metric: {', '.join(unknown)}")

        self.tpl = analyzer
        self.stats = analyzer.s
        self.tf = analyzer.timeframe
        self.seed = seed

        self._metric_funcs: Dict[MetricKey, Callable] = {
            m: (lambda pf, m=m: self._METRIC_MAPS[m](self.stats, self.tf, pf))
            for m in self.metrics
        }

    def finite_differences(
        self,
        base_params: Dict[str, Any],
        step_pct: float = 0.01,
    ) -> pd.DataFrame:
     
        f0 = self._objective(base_params)
        if not all(np.isfinite(list(f0.values()))):
            raise RuntimeError("Baseline metric not finite")

        rows = []
        for name, val in base_params.items():
            if isinstance(val, bool) or not isinstance(val, (int, float, np.floating)):
                continue 

            if isinstance(val, int):
                h = max(1, int(round(step_pct * max(abs(val), 1))))
                theta_minus, theta_plus = max(1, val - h), val + h
            else:
                h = step_pct * (abs(val) if val else 1.0)
                theta_minus, theta_plus = val - h, val + h

            p_minus = dict(base_params, **{name: theta_minus})
            p_plus  = dict(base_params, **{name: theta_plus})

            f_minus = self._objective(p_minus)
            f_plus  = self._objective(p_plus)

            row = dict(parameter=name, theta=val, h=h)
            for m in self.metrics:
                deriv = (f_plus[m] - f_minus[m]) / (2 * h)
                rel = (abs(deriv) * (abs(val) / abs(f0[m]))
                       if f0[m] not in (0, np.nan) else np.nan)
                row[f"derivative_{m}"] = deriv
                row[f"relative_sensitivity_{m}"] = rel
            rows.append(row)

        if not rows:
            raise ValueError("no numerical params found")

        return pd.DataFrame(rows).set_index("parameter")

    def _objective(self, params: Dict[str, Any]) -> Dict[MetricKey, float]:
        if self.seed is not None:
            state = np.random.get_state()
            np.random.seed(self.seed)

        A = self.tpl.__class__
        a = A(
            strategy=self.tpl.strategy,
            params=params,
            full_data=self.tpl.full_data,
            timeframe=self.tf,
            test_size=0.0,
            init_cash=self.tpl.init_cash,
            fees=self.tpl.fees,
            slippage=self.tpl.slippage,
            tp_stop=params.get("tp_pct"),
            sl_stop=params.get("sl_pct"),
        )

        vals = {m: func(a.pf) for m, func in self._metric_funcs.items()}

        if self.seed is not None:
            np.random.set_state(state)

        return vals

    def plot_finite_differences(
        self,
        matrix: Optional[pd.DataFrame] = None,
        step_pct: float = 0.01,
        title: Optional[str] = None,
    ) -> hv.HeatMap:
        if matrix is None:
            matrix = self.finite_differences(self.template.params, step_pct)
        title = title or f"(h={step_pct:.0%})"
    
        heatmap = _PlotFinDiff(matrix, title).heatmap()
        return heatmap
    
#