# evaluation.py
import pandas as pd
import numpy as np

def calculate_cumulative_return(portfolio_values):
    return portfolio_values[-1] / portfolio_values[0] - 1

def calculate_annualized_return(portfolio_values):
    daily_returns = pd.Series(portfolio_values).pct_change().dropna()
    return (1 + daily_returns.mean()) ** 252 - 1

def calculate_annualized_volatility(portfolio_values):
    daily_returns = pd.Series(portfolio_values).pct_change().dropna()
    return daily_returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.0):
    daily_returns = pd.Series(portfolio_values).pct_change().dropna()
    excess_return = daily_returns - risk_free_rate / 252
    return (excess_return.mean() / excess_return.std()) * np.sqrt(252)

def calculate_max_drawdown(portfolio_values):
    cumulative = pd.Series(portfolio_values)
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()

def evaluate_portfolio(portfolio_values, model_name="Agent"):
    cr = calculate_cumulative_return(portfolio_values)
    ar = calculate_annualized_return(portfolio_values)
    av = calculate_annualized_volatility(portfolio_values)
    sr = calculate_sharpe_ratio(portfolio_values)
    mdd = calculate_max_drawdown(portfolio_values)

    print(f"\n Evaluation Metrics for {model_name}:")
    print(f"  - Cumulative Return     : {cr:.4f}")
    print(f"  - Annualized Return     : {ar:.4f}")
    print(f"  - Annualized Volatility : {av:.4f}")
    print(f"  - Sharpe Ratio          : {sr:.4f}")
    print(f"  - Max Drawdown          : {mdd:.4f}")
    
    return {
        "Model": model_name,
        "Cumulative Return": cr,
        "Annualized Return": ar,
        "Annualized Volatility": av,
        "Sharpe Ratio": sr,
        "Max Drawdown": mdd,
    }
