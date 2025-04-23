# QuantyBT 🪐

**A lightweight backtesting framework based on [vectorbt](https://github.com/polakowo/vectorbt), focused on statistical robustness, modularity, and seamless strategy integration with custom models and crypto-native data loading.**  
---

## Features

- **Simple integration** with vectorbt as the backtesting engine (`bt_instance`).
- **Custom Model Support**: Native wrappers for custom-implemented models (e.g., Kalman Filters) and statistical frameworks.  
- **Built-in data loaders** for cryptocurrencies from Binance (no api needed!).
- **Modular architecture**: define strategies by inheriting from a base `Strategy` class (`preprocess`, `generate_signals`, `param_space`).
- **Robust Validation**: Out-of-sample splits and hyperparameter tuning via [Hyperopt](https://github.com/hyperopt/hyperopt).  
- **Statistical analysis tools**: Monte Carlo simulations
- **Performance reporting**: generate equity curves, heatmaps, and metric summaries with minimal boilerplate.

---

## Incoming Features

- **More Custom Models** 
- **Walk-Forward Optimization (WFO)** with automated plotting and summary reports  
- **Sensitivity Analysis** for identifying and mitigating overfitting  
- **Portfolio Optimization** with advanced methods (HRP, CVaR, Maximum Entropy, ...)  
- **Live Execution via CCXT**: Seamless end-to-end workflow from strategy design → testing → deployment.

---

## Installation

Install the package via pip:

```bash
pip install quantybt

```

