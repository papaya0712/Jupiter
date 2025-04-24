# quantybt/montecarlo.py
# quantybt/montecarlo.py
import numpy as np
import pandas as pd
from typing import Optional, Any, Dict, List

try:
    from numba import njit

    @njit(cache=True, fastmath=True)
    def _cumprod_numba(a: np.ndarray) -> np.ndarray:
        out = np.empty_like(a)
        for i in range(a.shape[0]):
            acc = 1.0
            for j in range(a.shape[1]):
                acc *= a[i, j]
                out[i, j] = acc
        return out
except Exception: 
    def _cumprod_numba(a: np.ndarray) -> np.ndarray:  
        return np.cumprod(a, axis=1)
# --------------------------------------------------------------------------- #

class MonteCarloBootstrapping:
    _PERIODS = {
        '1m': 525_600, '5m': 105_120, '15m': 35_040, '30m': 17_520,
        '1h': 8_760,   '2h': 4_380,   '4h': 2_190,
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

    def _convert_frequency(self, ret: pd.Series) -> pd.Series:
        rs = ret.copy()
        rs.index = pd.to_datetime(rs.index)

        if self.timeframe.endswith(('m', 'h')) or self.timeframe == '1d':
            return rs
        if self.timeframe == '1w':
            return rs.resample('W').apply(lambda x: (1 + x).prod() - 1)
        return rs.resample('M').apply(lambda x: (1 + x).prod() - 1)

    def _analyze_simulations(
        self, samples: np.ndarray
    ) -> List[Dict[str, float]]:
     
        ann_factor = self.ann_factor
        init_cash = self.init_cash
        cumprod = _cumprod_numba(1.0 + samples) * init_cash
        cum_ret = (1.0 + samples).prod(axis=1) - 1.0
        std = samples.std(axis=1, ddof=1)
        ann_vol = std * np.sqrt(ann_factor)
        mean_ret = samples.mean(axis=1)
        sharpe = np.where(std > 0, mean_ret / std * np.sqrt(ann_factor), np.nan)
        rolling_max = np.maximum.accumulate(cumprod, axis=1)
        max_dd = ((cumprod - rolling_max) / rolling_max).min(axis=1)

        out = []
        for i in range(samples.shape[0]):
            out.append({
                'CumulativeReturn': cum_ret[i],
                'AnnVol':          ann_vol[i],
                'Sharpe':          sharpe[i],
                'MaxDrawdown':     max_dd[i]
            })
        return out

    def mc_with_replacement(self) -> Dict[str, Any]:
        np.random.seed(self.random_seed)

        returns = self._convert_frequency(self.ret_series)
        arr = returns.values.astype(np.float64)
        n_obs = arr.size

     
        idx = np.random.randint(0, n_obs, size=(self.n_sims, n_obs))
        samples = arr[idx]

        equity = _cumprod_numba(1.0 + samples) * self.init_cash

        sim_equity = pd.DataFrame(
            equity.T,               
            index=returns.index,
            columns=[f"Sim_{i}" for i in range(self.n_sims)]
        )

        sim_stats = self._analyze_simulations(samples)
        orig_stats = self._analyze_simulations(arr[np.newaxis, :])[0]

        return {
            'original_stats':          orig_stats,
            'simulated_stats':         sim_stats,
            'simulated_equity_curves': sim_equity
        }

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

    def plot(self):
        from quantybt.plots import _PlotBootstrapping
        return _PlotBootstrapping(self).plot()
