# quantybt/montecarlo.py
import pandas as pd
import numpy as np
from typing import Optional, Any, Dict, Tuple
from matplotlib.figure import Figure

class MonteCarloBootstrapping:
    _PERIODS = {
        '1m': 525600, '5m': 105120, '15m': 35040, '30m': 17520,
        '1h': 8760,   '2h': 4380,   '4h': 2190,  
        '1d': 365, '1w': 52
        }

    def __init__(
        self,
        analyzer: Optional[Any] = None,
        *,
        timeframe: str = '1d',
        ret_series: Optional[pd.Series] = None,
        n_sims: int = 250,
        random_seed: int = 69
    ):
        if analyzer is not None:
            self.pf = analyzer.pf
            self.init_cash = analyzer.init_cash
            self.timeframe = analyzer.timeframe
            self.ret_series = analyzer.pf.returns()
        else:
            if ret_series is None:
                raise ValueError("Provide a return series if no analyzer is given")
            self.pf = None
            self.init_cash = 1.0
            self.timeframe = timeframe
            self.ret_series = ret_series.copy()

        if self.timeframe not in self._PERIODS:
            raise ValueError(f"Unsupported timeframe '{self.timeframe}'.")

        self.n_sims = n_sims
        self.random_seed = random_seed
        self.ann_factor = self._PERIODS[self.timeframe]

    def _compute_drawdown(self, equity: pd.Series) -> pd.Series:
        rolling_max = equity.cummax()
        return (equity - rolling_max) / rolling_max

    def _convert_frequency(self, ret_series: pd.Series) -> pd.Series:
        rs = ret_series.copy()
        rs.index = pd.to_datetime(rs.index)
        if self.timeframe.endswith(('m', 'h')) or self.timeframe == '1d':
            return rs
        if self.timeframe == '1w':
            return rs.resample('W').apply(lambda x: (1 + x).prod() - 1)
        return rs.resample('M').apply(lambda x: (1 + x).prod() - 1)

    def _analyze_series(self, ret: pd.Series) -> Dict[str, float]:
        if len(ret) < 2:
            return dict.fromkeys(['CumulativeReturn','AnnVol','Sharpe','MaxDrawdown'], np.nan)
        cumret = (1 + ret).prod() - 1
        vol = ret.std(ddof=1) * np.sqrt(self.ann_factor)
        mean_ret, std_ret = ret.mean(), ret.std(ddof=1)
        sharpe = (mean_ret / std_ret) * np.sqrt(self.ann_factor) if std_ret else np.nan
        equity = (1 + ret).cumprod() * self.init_cash
        max_dd = self._compute_drawdown(equity).min()
        return {
            'CumulativeReturn': cumret,
            'AnnVol': vol,
            'Sharpe': sharpe,
            'MaxDrawdown': max_dd
        }

    def mc_with_replacement(self) -> Dict[str, Any]:
        np.random.seed(self.random_seed)
        returns = self._convert_frequency(self.ret_series)
        arr = returns.values
        n = len(arr)
        all_equities, sim_stats = [], []
        for i in range(self.n_sims):
            idx = np.random.choice(n, size=n, replace=True)
            sample = pd.Series(arr[idx], index=returns.index)
            equity = (1 + sample).cumprod() * self.init_cash
            all_equities.append(equity)
            sim_stats.append(self._analyze_series(sample))
        sim_equity = pd.concat([eq.rename(f"Sim_{i}") for i, eq in enumerate(all_equities)], axis=1)
        orig_stats = self._analyze_series(returns)
        return {'original_stats': orig_stats, 'simulated_stats': sim_stats, 'simulated_equity_curves': sim_equity}

    def benchmark_equity(self) -> pd.Series:
        if self.pf is not None and hasattr(self.pf, 'benchmark_value'):
            bench = self.pf.benchmark_value()
        else:
            orig_ret = self._convert_frequency(self.ret_series)
            bench = (1 + orig_ret).cumprod() * self.init_cash
        bench.index = pd.to_datetime(bench.index)
        return bench

    def results(self) -> pd.DataFrame:
        res = self.mc_with_replacement()
        df = pd.DataFrame(res['simulated_stats'])
        df.loc['Original'] = res['original_stats']
        return df

    def plot(self) -> Figure:
        from quantybt.plots import _PlotBootstrapping
        plotter = _PlotBootstrapping(self)
        return plotter.plot()
