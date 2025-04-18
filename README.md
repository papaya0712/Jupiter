# QuantyBT 🪐

**A lightweight extension for [`vectorbt`](https://github.com/polakowo/vectorbt) focused on statistical robustness, modularity, and seamless strategy integration.**

QuantyBT combines the familiar VectorBT API for fast in‑sample backtests with a suite of built‑in, advanced validation techniques. Define your strategy exactly as before—implementing `preprocess_data`, `generate_signals`, and `param_space`—and then unlock:

- **Out‑of‑Sample Splits** for rigorous train/test separation  
- **Walk‑Forward Optimization** to guard against look‑ahead bias  
- **Monte Carlo Simulations** for probabilistic performance envelopes  
- **Parameter Sensitivity Analyses** to pinpoint robust hyperparameter regions  

All features plug seamlessly into your existing VectorBT workflows, helping you minimize overfitting and boost the real‑world robustness of your trading algorithms—without rewriting a single line of strategy code.

---

## 🚀 Features

- Clean `Strategy` base class for building reusable strategies  
- `Analyzer` with detailed performance metrics (Sharpe, Sortino, Calmar, volatility, drawdown, etc.)  
- Time‑based train/test split with purge window  
- Built‑in `hyperopt` integration for parameter optimization  
- Modular structure—easy to extend, simple to use  

---

## 🔬 Coming Soon

- Full Walk‑Forward Optimization  
- Monte Carlo simulations (bootstrapping, permutation, resampling)  
- Hyperparameter sensitivity analysis (“stress tests”)  
- Additional statistical tests and metrics  
- Expanded portfolio management features  

---

## 📦 Installation

```bash
pip install quantybt
```

## 📖 Documentation & Examples

See the `examples/` folder for Jupyter notebooks demonstrating:

- Strategy implementation and backtesting  
- In‑Sample vs. Out‑of‑Sample comparison  
- Hyperparameter optimization with Hyperopt  
- Monte Carlo and sensitivity analyses  

## ✨ Contributing & Feedback

Contributions, issues, and feature requests are welcome!  
If you use QuantyBT in your work or research, a mention or link back to this repo is greatly appreciated.  

## 📄 License

This project is released under the MIT License. See [LICENSE](./LICENSE-txt) for details.  
