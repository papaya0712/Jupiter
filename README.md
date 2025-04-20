# QuantyBT 🪐

**A lightweight backtesting framework based on [vectorbt](https://github.com/polakowo/vectorbt) focused on statistical robustness, modularity, and seamless strategy integration with custom-implemented models and crypto focused data-loader.**

---

## Features

- **Simple integration** with vectorbt as the backtesting engine (`bt_instance`).
- **Custom model support**: native wrappers for Hawkes processes, Bayesian models, and other statistical frameworks.
- **Built-in data loaders** for cryptocurrencies (e.g., Bitcoin, Ethereum).
- **Modular architecture**: define strategies by inheriting from a base `Strategy` class (`preprocess`, `generate_signals`, `param_space`).
- **Robust validation**: out-of-sample splits, walk-forward optimization, and hyperparameter tuning via `hyperopt`.
- **Statistical analysis tools**: Monte Carlo simulations, bootstrapping of trade outcomes, and sensitivity analysis.
- **Performance reporting**: generate equity curves, heatmaps, and metric summaries with minimal boilerplate.

---

## Installation

Install the package via pip:

```bash
pip install quantybt

```

