"""Volatility forecasting & VaR backtesting framework.

Modules
-------
data        : download / cache prices, compute log returns, build portfolio
volatility  : rolling, EWMA and GARCH(1,1) volatility estimators
var         : VaR / Expected-Shortfall calculators and the walk-forward engine
backtest    : exceedance counting, Kupiec POF, Christoffersen, Basel traffic light
"""

__all__ = ["data", "volatility", "var", "backtest"]
