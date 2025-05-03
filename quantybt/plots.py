import pandas as pd
import numpy as np

import holoviews as hv
import plotly.graph_objects as go

from typing import Tuple, TYPE_CHECKING
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

#### ============= normal Backtest Summary ============= ####
class _PlotBacktest:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.pf = analyzer.pf
        self.s = analyzer.s

    def plot_backtest(
        self,
        title: str = "Backtest Results",
        export_html: bool = False,
        export_image: bool = False,
        file_name: str = "backtest_plot[QuantyBT]",
    ) -> go.Figure:
        strategy_equity = self.pf.value()
        try:
            benchmark_equity = self.pf.benchmark_value()
        except AttributeError:
            benchmark_equity = pd.Series(index=strategy_equity.index, dtype=float)

        strat_dd = (
            (strategy_equity - strategy_equity.cummax()) / strategy_equity.cummax() * 100
        )
        bench_dd = (
            (benchmark_equity - benchmark_equity.cummax()) / benchmark_equity.cummax() * 100
            if not benchmark_equity.empty
            else pd.Series(index=strategy_equity.index, dtype=float)
        )

        rets = self.pf.returns()

        trades = self.pf.trades.records_readable
        entries = trades["Entry Timestamp"].astype("int64")
        exits = trades["Exit Timestamp"].fillna(strategy_equity.index[-1]).astype("int64")
        idx_int = rets.index.astype("int64").values
        open_trades = (
            (idx_int[:, None] >= entries.values) & (idx_int[:, None] <= exits.values)
        ).any(axis=1)
        rets = rets[open_trades]

        factor_root = self.s._annual_factor(self.analyzer.timeframe, root=True)
        factor = self.s._annual_factor(self.analyzer.timeframe, root=False)
        window = max(1, int(factor / 2))
        window_label = "180d"

        strat_mean = rets.rolling(window, min_periods=window).mean()
        strat_std = rets.rolling(window, min_periods=window).std(ddof=1)
        rolling_sharpe = (strat_mean / strat_std) * factor_root

        try:
            bench_rets = self.pf.benchmark_returns()
            bench_mean = bench_rets.rolling(window, min_periods=window).mean()
            bench_std = bench_rets.rolling(window, min_periods=window).std(ddof=1)
            rolling_bench_sharpe = (bench_mean / bench_std) * factor_root
        except AttributeError:
            rolling_bench_sharpe = pd.Series(index=rolling_sharpe.index, dtype=float)

        rolling_sharpe = rolling_sharpe.iloc[window:]
        rolling_bench_sharpe = rolling_bench_sharpe.iloc[window:]

        if "Return [%]" in trades.columns:
            trade_returns = (
                trades["Return [%]"].astype(str).str.rstrip("% ").astype(float)
            )
        else:
            trade_returns = trades["Return"].dropna() * 100

        kde = gaussian_kde(trade_returns.values, bw_method="scott")
        x_kde = np.linspace(trade_returns.min(), trade_returns.max(), 200)
        y_kde = kde(x_kde) * 100

        fig = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=False,
            vertical_spacing=0.05,
            horizontal_spacing=0.1,
            row_heights=[0.5, 0.5],
            column_widths=[0.7, 0.3],
            subplot_titles=[
                "Equity Curve",
                "Rolling Sharpe",
                "Drawdown Curve",
                "Trade Returns Distribution",
            ],
        )

        fig.add_trace(
            go.Scatter(
                x=strategy_equity.index,
                y=strategy_equity.values,
                mode="lines",
                name="Strategy Equity",
                #fill="tozeroy",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=benchmark_equity.index,
                y=benchmark_equity.values,
                mode="lines",
                name="Benchmark Equity",
                #fill="tozeroy",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode="lines",
                name=f"Rolling Sharpe (Strategy) ({window_label})",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=rolling_bench_sharpe.index,
                y=rolling_bench_sharpe.values,
                mode="lines",
                name=f"Rolling Sharpe (Benchmark) ({window_label})",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line=dict(color="white", dash="dash", width=2), row=1, col=2)

        fig.add_trace(
            go.Scatter(
                x=strategy_equity.index,
                y=strat_dd.values,
                mode="lines",
                name="Strategy Drawdown",
                fill="tozeroy",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=benchmark_equity.index,
                y=bench_dd.values,
                mode="lines",
                name="Benchmark Drawdown",
                fill="tozeroy",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=trade_returns,
                nbinsx=50,
                histnorm="percent",
                name="Return Histogram",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=x_kde, y=y_kde, mode="lines", name="KDE (%)"),
            row=2,
            col=2,
        )
        fig.add_vline(x=0, line=dict(color="white", dash="dash", width=2), row=2, col=2)

        fig.update_layout(
            title=title, hovermode="x unified", template="plotly_dark", height=700
        )
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(title_text="Returns [%]", row=2, col=2)

        if export_html:
            fig.write_html(f"{file_name}.html")
        if export_image:
            try:
                fig.write_image(f"{file_name}.png")
            except ValueError:
                pass

        return fig
    
#### ============= opt Summary ============= ####
class _PlotTrainTestSplit:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.analyzer = optimizer.analyzer
        self.s = self.analyzer.s

    def plot_oos(self,
                 title: str = 'In-Sample vs Out-of-Sample Performance',
                 export_html: bool = False,
                 export_image: bool = False,
                 file_name: str = 'train_test_plot[QuantyBT]') -> go.Figure:
        
        eq_train = self.optimizer.train_pf.value()
        eq_test  = self.optimizer.test_pf.value()

        # Drawdowns
        dd_train = self.optimizer.train_pf.drawdown()
        dd_test  = self.optimizer.test_pf.drawdown()

        # metrics
        metrics = ['CAGR [%]', 
                   'Max Drawdown (%)',
                   'Sharpe Ratio', 
                   'Sortino Ratio', 
                   'Calmar Ratio']
        
        train_metrics = self.s.backtest_summary(self.optimizer.train_pf, self.analyzer.timeframe)
        test_metrics  = self.s.backtest_summary(self.optimizer.test_pf, self.analyzer.timeframe)

        train_vals = [
            train_metrics.loc['CAGR [%]', 'Value'],
            abs(train_metrics.loc['Strategy Max Drawdown [%]', 'Value']),
            train_metrics.loc['Sharpe Ratio', 'Value'],
            train_metrics.loc['Sortino Ratio', 'Value'],
            train_metrics.loc['Calmar Ratio', 'Value']
        ]
        test_vals = [
            test_metrics.loc['CAGR [%]', 'Value'],
            abs(test_metrics.loc['Strategy Max Drawdown [%]', 'Value']),
            test_metrics.loc['Sharpe Ratio', 'Value'],
            test_metrics.loc['Sortino Ratio', 'Value'],
            test_metrics.loc['Calmar Ratio', 'Value']
        ]

        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "xy"}, {"type": "table"}],
                   [{"type": "xy", "colspan": 2}, None]],
            subplot_titles=['Equity Curves', 'Metrics Comparison', 'Drawdown Curves [%]'],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )

        
        is_color, oos_color = "#2ecc71", "#3498db"
        is_fill, oos_fill   = "rgba(46, 204, 113, 0.2)", "rgba(52, 152, 219, 0.2)"

        # Equity Traces
        fig.add_trace(go.Scatter(x=eq_train.index, y=eq_train.values, mode='lines',
                                 name='In-Sample Equity', line=dict(color=is_color)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=eq_test.index, y=eq_test.values, mode='lines',
                                 name='Out-of-Sample Equity', line=dict(color=oos_color)),
                      row=1, col=1)

        # table
        fig.add_trace(go.Table(
            header=dict(values=['Metric', 'IS', 'OOS']),
            cells=dict(values=[metrics, train_vals, test_vals])
        ), row=1, col=2)

        n1 = len(dd_train)
        x_train = np.arange(n1)
        x_test  = np.arange(n1, n1 + len(dd_test))

        fig.add_trace(
            go.Scatter(
                x=x_train,
                y=dd_train.values,
                mode="lines",
                name="In-Sample Drawdown",
                line=dict(color=is_color),  
                fill="tozeroy",
                fillcolor=is_fill
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_test,
                y=dd_test.values,
                mode="lines",
                name="Out-of-Sample Drawdown",
                line=dict(color=oos_color), 
                fill="tozeroy",
                fillcolor=oos_fill
            ),
            row=2, col=1
        )

        fig.update_layout(title=title, height=800, showlegend=True, template="plotly_dark")

        if export_html:
            fig.write_html(f"{file_name}.html")
        if export_image:
            try:
                fig.write_image(f"{file_name}.png")
            except ValueError:
                pass

        return fig

class _PlotGeneralization:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def plot_generalization(self, title: str = "IS vs OOS Performance") -> hv.Layout:
        hv.extension('bokeh')
        hv.renderer('bokeh').theme = 'carbon'
        
        if not self.optimizer.trial_metrics:
            raise ValueError("No trial metrics found. Run optimizer.optimize() first.")

        trial_metrics = np.array(self.optimizer.trial_metrics)
        unique_metrics = np.unique(trial_metrics, axis=0)
        is_scores = unique_metrics[:, 0]
        oos_scores = unique_metrics[:, 1]

        coeffs = np.polyfit(is_scores, oos_scores, deg=1)
        fit_line = np.poly1d(coeffs)
        x_fit = np.linspace(min(is_scores.min(), oos_scores.min()), max(is_scores.max(), oos_scores.max()), 100)
        y_fit = fit_line(x_fit)

        y_true = oos_scores
        y_pred = fit_line(is_scores)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        title = f"{title} (R² = {r2:.2f})"

        scatter = hv.Scatter((is_scores, oos_scores), 'IS Performance', 'OOS Performance').opts(
            size=6,
            color='deepskyblue',
            tools=['hover'],
            xlabel='In-Sample',
            ylabel='Out-of-Sample',
            width=600,
            height=600,
            title=title,
            show_grid=True,
            bgcolor=None,
            gridstyle={'grid_line_alpha': 0.3},
        )

        regression = hv.Curve((x_fit, y_fit), label=f"Linear Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}").opts(
            color='orange',
            line_width=2,
            line_dash='dashed'
        )

        ideal = hv.Curve((x_fit, x_fit), label="Ideal 45° Line").opts(
            color='lightgrey',
            line_width=2,
            line_dash='dotted'
        )

        layout = (scatter * regression * ideal).opts(
            hv.opts.Overlay(legend_position='bottom_right', shared_axes=True)
        )

        return layout
    
#### ============= Montecarlo Bootstrapping Summary ============= ####
if TYPE_CHECKING:
    from quantybt.montecarlo import MonteCarloBootstrapping

class _PlotBootstrapping:
    def __init__(self, mc):
        self.mc = mc

    def _align_series(self, sim_eq: pd.DataFrame, bench_eq: pd.Series):
        sim_eq.index = pd.to_datetime(sim_eq.index)
        bench_eq.index = pd.to_datetime(bench_eq.index)
        start = max(sim_eq.index.min(), bench_eq.index.min())
        end = min(sim_eq.index.max(), bench_eq.index.max())
        sim_eq, bench_eq = sim_eq.loc[start:end], bench_eq.loc[start:end]
        idx = sim_eq.index.union(bench_eq.index)
        return sim_eq.reindex(idx).ffill(), bench_eq.reindex(idx).ffill()

    def plot_histograms(self, mc_results: pd.DataFrame = None):
        hv.extension('bokeh')
        hv.renderer('bokeh').theme = 'carbon'

        if mc_results is None:
            data = self.mc.mc_with_replacement()
            mc_results = pd.DataFrame(data['simulated_stats'])

        sharpe_vals = mc_results['Sharpe']
        sortino_vals = mc_results['Sortino']
        calmar_vals = mc_results['Calmar']
        maxdd_vals = mc_results['MaxDrawdown']

        sharpe_q5, sharpe_q50, sharpe_q95 = np.percentile(sharpe_vals, [5, 50, 95])
        sortino_q5, sortino_q50, sortino_q95 = np.percentile(sortino_vals, [5, 50, 95])
        calmar_q5, calmar_q50, calmar_q95 = np.percentile(calmar_vals, [5, 50, 95])
        maxdd_q5, maxdd_q50, maxdd_q95 = np.percentile(maxdd_vals, [5, 50, 95])

        bench_ret = self.mc.pf.benchmark_returns()
        bench_stats = self.mc._analyze_series(bench_ret)

        bench_sharpe = bench_stats['Sharpe']
        bench_sortino = bench_stats['Sortino']
        bench_calmar = bench_stats['Calmar']
        bench_maxdd = bench_stats['MaxDrawdown']

        color_q5 = "green"
        color_q50 = "deepskyblue"
        color_q95 = "red"
        color_bench = "purple"
        bins = 50
        plot_width = 600
        plot_height = 400

        hist_opts = dict(fill_color="lightgrey", bgcolor=None, gridstyle={'grid_line_alpha': 0.3}, width=plot_width, height=plot_height, show_legend=False)

        sharpe_vals = sharpe_vals[np.isfinite(sharpe_vals)]
        sortino_vals = sortino_vals[np.isfinite(sortino_vals)]
        calmar_vals = calmar_vals[np.isfinite(calmar_vals)]
        maxdd_vals = maxdd_vals[np.isfinite(maxdd_vals)]

        sharpe_hist_values, sharpe_bin_edges = np.histogram(sharpe_vals, bins=bins)
        sortino_hist_values, sortino_bin_edges = np.histogram(sortino_vals, bins=bins)
        calmar_hist_values, calmar_bin_edges = np.histogram(calmar_vals, bins=bins)
        maxdd_hist_values, maxdd_bin_edges = np.histogram(maxdd_vals, bins=bins)


        hist_sharpe = hv.Histogram((sharpe_hist_values, sharpe_bin_edges)).opts(title="Sharpe Distribution", xlabel="Sharpe", ylabel="Frequency", **hist_opts)
        hist_sortino = hv.Histogram((sortino_hist_values, sortino_bin_edges)).opts(title="Sortino Distribution", xlabel="Sortino", ylabel="Frequency", **hist_opts)
        hist_calmar = hv.Histogram((calmar_hist_values, calmar_bin_edges)).opts(title="Calmar Distribution", xlabel="Calmar", ylabel="Frequency", **hist_opts)
        hist_maxdd = hv.Histogram((maxdd_hist_values, maxdd_bin_edges)).opts(title="Max Drawdown Distribution", xlabel="MaxDrawdown", ylabel="Frequency", **hist_opts)

        max_y_sharpe = sharpe_hist_values.max()
        max_y_sortino = sortino_hist_values.max()
        max_y_calmar = calmar_hist_values.max()
        max_y_maxdd = maxdd_hist_values.max()

        spikes_sharpe = (
            hv.Spikes(pd.DataFrame({'Sharpe': [sharpe_q5], 'y': [max_y_sharpe]}), kdims='Sharpe', vdims='y', label='5th %ile').opts(color=color_q5, line_dash="dashed", line_width=2) *
            hv.Spikes(pd.DataFrame({'Sharpe': [sharpe_q50], 'y': [max_y_sharpe]}), kdims='Sharpe', vdims='y', label='50th %ile').opts(color=color_q50, line_dash="solid", line_width=2) *
            hv.Spikes(pd.DataFrame({'Sharpe': [sharpe_q95], 'y': [max_y_sharpe]}), kdims='Sharpe', vdims='y', label='95th %ile').opts(color=color_q95, line_dash="dashed", line_width=2))

        spikes_sortino = (
            hv.Spikes(pd.DataFrame({'Sortino': [sortino_q5], 'y': [max_y_sortino]}), kdims='Sortino', vdims='y', label='5th %ile').opts(color=color_q5, line_dash="dashed", line_width=2) *
            hv.Spikes(pd.DataFrame({'Sortino': [sortino_q50], 'y': [max_y_sortino]}), kdims='Sortino', vdims='y', label='50th %ile').opts(color=color_q50, line_dash="solid", line_width=2) *
            hv.Spikes(pd.DataFrame({'Sortino': [sortino_q95], 'y': [max_y_sortino]}), kdims='Sortino', vdims='y', label='95th %ile').opts(color=color_q95, line_dash="dashed", line_width=2))

        spikes_calmar = (
            hv.Spikes(pd.DataFrame({'Calmar': [calmar_q5], 'y': [max_y_calmar]}), kdims='Calmar', vdims='y', label='5th %ile').opts(color=color_q5, line_dash="dashed", line_width=2) *
            hv.Spikes(pd.DataFrame({'Calmar': [calmar_q50], 'y': [max_y_calmar]}), kdims='Calmar', vdims='y', label='50th %ile').opts(color=color_q50, line_dash="solid", line_width=2) *
            hv.Spikes(pd.DataFrame({'Calmar': [calmar_q95], 'y': [max_y_calmar]}), kdims='Calmar', vdims='y', label='95th %ile').opts(color=color_q95, line_dash="dashed", line_width=2))

        spikes_maxdd = (
            hv.Spikes(pd.DataFrame({'MaxDrawdown': [maxdd_q5], 'y': [max_y_maxdd]}), kdims='MaxDrawdown', vdims='y', label='5th %ile').opts(color=color_q5, line_dash="dashed", line_width=2) *
            hv.Spikes(pd.DataFrame({'MaxDrawdown': [maxdd_q50], 'y': [max_y_maxdd]}), kdims='MaxDrawdown', vdims='y', label='50th %ile').opts(color=color_q50, line_dash="solid", line_width=2) *
            hv.Spikes(pd.DataFrame({'MaxDrawdown': [maxdd_q95], 'y': [max_y_maxdd]}), kdims='MaxDrawdown', vdims='y', label='95th %ile').opts(color=color_q95, line_dash="dashed", line_width=2))

        bench_sh_spike = hv.Spikes(pd.DataFrame({'Sharpe': [bench_sharpe], 'y': [max_y_sharpe]}), kdims='Sharpe', vdims='y', label='Benchmark').opts(color=color_bench, line_dash="solid", line_width=2)
        bench_so_spike = hv.Spikes(pd.DataFrame({'Sortino': [bench_sortino], 'y': [max_y_sortino]}), kdims='Sortino', vdims='y', label='Benchmark').opts(color=color_bench, line_dash="solid", line_width=2)
        bench_ca_spike = hv.Spikes(pd.DataFrame({'Calmar': [bench_calmar], 'y': [max_y_calmar]}), kdims='Calmar', vdims='y', label='Benchmark').opts(color=color_bench, line_dash="solid", line_width=2)
        bench_dd_spike = hv.Spikes(pd.DataFrame({'MaxDrawdown': [bench_maxdd], 'y': [max_y_maxdd]}), kdims='MaxDrawdown', vdims='y', label='Benchmark').opts(color=color_bench, line_dash="solid", line_width=2)

        plot_sharpe = (hist_sharpe * spikes_sharpe * bench_sh_spike).opts(show_legend=True, legend_position='top_right')
        plot_sortino = (hist_sortino * spikes_sortino * bench_so_spike).opts(show_legend=True, legend_position='top_right')
        plot_calmar = (hist_calmar * spikes_calmar * bench_ca_spike).opts(show_legend=True, legend_position='top_right')
        plot_maxdd = (hist_maxdd * spikes_maxdd * bench_dd_spike).opts(show_legend=True, legend_position='top_right')

        final_plot = (plot_sharpe + plot_sortino + plot_calmar + plot_maxdd).opts(shared_axes=False)
        return final_plot
    
#### ============= Finite Differences analysis Summary ============= #### 
class _PlotFinDiff:
    def __init__(self, matrix: pd.DataFrame, title: str = "Parameter Sensitivities"):
        self.df = matrix
        self.title = title

    def heatmap(self) -> hv.HeatMap:
        data = (
            self.df.filter(like="relative_sensitivity_")
                .rename(columns=lambda c: c.replace("relative_sensitivity_", ""))
                .reset_index()
                .melt(id_vars="parameter", var_name="metric", value_name="rs")
        )

        return hv.HeatMap(
            data,
            kdims=["parameter", "metric"],
            vdims=["rs"]
        ).opts(
            cmap="Plasma",
            colorbar=True,
            invert_yaxis=True,
            width=600,
            height=300,
            tools=["hover"],
            xlabel="Metrik",
            ylabel="Parameter",
            hover_tooltips=[
                ("Parameter", "@parameter"),
                ("Metrik", "@metric"),
                ("RelSens", "@rs{0.000}"),
            ],
            title=self.title,
        )

#