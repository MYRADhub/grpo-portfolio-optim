import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler
import torch

from stable_baselines3 import PPO
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler, data_split
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.main import check_and_make_directories
from grpo import GRPOAgent

# ---------- Ensure Output Directories Exist ----------
# The training scripts create these, but plots.py is documented as a standalone
# entry point and must not depend on a training run having happened first.
check_and_make_directories(["results", "trained_models"])

# ---------- Prepare Test Data ----------
tickers = ["^GSPC", "^GDAXI", "^IXIC", "^RUT", "^N225"]
df_raw = YahooDownloader(start_date='1988-02-01', end_date='2024-04-30', ticker_list=tickers).fetch_data()
df_raw = df_raw.pivot(index="date", columns="tic", values=["open", "high", "low", "close", "volume"])
df_raw = df_raw.ffill().dropna()
df_raw = df_raw.stack(level="tic").reset_index()
df_norm = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(df_raw)
df_portfolio = df_norm[["date", "tic", "close", "high", "low"]]
df_test = data_split(df_portfolio, '2016-01-01', '2024-04-30')

env_kwargs = {
    "initial_amount": 100000,
    "comission_fee_pct": 0.0025,
    "time_window": 50,
    "features": ["close", "high", "low"],
    "normalize_df": None
}

# ---------- PPO ----------
env_test_ppo = PortfolioOptimizationEnv(df=df_test, **env_kwargs)
ppo_model = PPO.load("trained_models/ppo_portfolio")
ppo_values = [env_test_ppo._initial_amount]
state = env_test_ppo.reset()
done = False
while not done:
    action, _ = ppo_model.predict(state, deterministic=True)
    action = np.clip(action, 0, 1)
    action /= np.sum(action) if np.sum(action) > 0 else 1
    state, _, done, _ = env_test_ppo.step(action)
    ppo_values.append(env_test_ppo._portfolio_value)

# ---------- PG (EIIE) ----------
env_test_pg = PortfolioOptimizationEnv(df=df_test, **env_kwargs)
pg_agent = DRLAgent(env_test_pg)
model_pg = pg_agent.get_model("pg", model_kwargs={"lr": 0.0003, "policy": EIIE}, policy_kwargs={"time_window": 50, "k_size": 3})
model_pg.train_policy.load_state_dict(torch.load("trained_models/pg_eiie.pt"))
DRLAgent.DRL_validation(model_pg, env_test_pg)
pg_values = env_test_pg._asset_memory["final"]

# ---------- GRPO ----------
env_test_grpo = PortfolioOptimizationEnv(df=df_test, **env_kwargs)
grpo_agent = GRPOAgent(env_test_grpo, policy_kwargs={"hidden_sizes": [64, 64]}, lr=0.0003, gamma=0.99, group_size=4, epsilon=0.15, beta=0.0005)
grpo_agent.policy.load_state_dict(torch.load("trained_models/grpo_policy.pt"))

state = env_test_grpo.reset()
done = False
grpo_values = [env_test_grpo._initial_amount]
while not done:
    action = grpo_agent.predict(state)
    state, _, done, _ = env_test_grpo.step(action)
    grpo_values.append(env_test_grpo._portfolio_value)

def compute_daily_returns(values):
    return pd.Series(values).pct_change().fillna(0)

def compute_market_baseline(df_test, time_window=50, initial_amount=100000):
    # Filter for S&P500 only
    sp500_df = df_test[df_test["tic"] == "^GSPC"]
    
    # Sort by date and set index
    sp500_df = sp500_df.sort_values("date")
    sp500_df["date"] = pd.to_datetime(sp500_df["date"])
    sp500_prices = sp500_df.set_index("date")["close"]

    # Calculate market portfolio value
    returns = sp500_prices.pct_change().fillna(0)
    market_value = (returns + 1).cumprod() * initial_amount

    # Skip first `time_window` steps (align with env start)
    return market_value.iloc[time_window:].tolist()

def smooth_series(series, window=10):
    return pd.Series(series).rolling(window=window, min_periods=1).mean().tolist()

def compute_bounds(series, window=10):
    s = pd.Series(series)
    min_series = s.rolling(window=window, min_periods=1).min().tolist()
    max_series = s.rolling(window=window, min_periods=1).max().tolist()
    return min_series, max_series

def compute_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess_returns = pd.Series(returns) - risk_free_rate / periods_per_year
    mean = excess_returns.mean()
    std = excess_returns.std()
    if std == 0:
        return 0.0
    sharpe = (mean / std) * np.sqrt(periods_per_year)
    return sharpe

def compute_annualized_volatility(returns, periods_per_year=252):
    return pd.Series(returns).std() * np.sqrt(periods_per_year)

def compute_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    returns = pd.Series(returns)
    downside_returns = returns[returns < 0]
    expected_return = returns.mean() - risk_free_rate / periods_per_year
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0
    return (expected_return * periods_per_year) / (downside_std * np.sqrt(periods_per_year))

def compute_max_drawdown(values):
    cumulative = pd.Series(values)
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def compute_calmar_ratio(values):
    annual_return = compute_annualized_return(values)
    max_drawdown = abs(compute_max_drawdown(values))
    return annual_return / max_drawdown if max_drawdown > 0 else 0.0

def compute_annualized_return(values, periods_per_year=252):
    values = pd.Series(values)
    total_periods = len(values) - 1
    if total_periods <= 0 or values.iloc[0] == 0:
        return 0.0
    total_return = values.iloc[-1] / values.iloc[0]
    annualized = total_return ** (periods_per_year / total_periods) - 1
    return annualized

def compute_final_value(values):
    return values[-1] if len(values) > 0 else 0.0

# Apply smoothing to portfolio values
grpo_values_smooth = smooth_series(grpo_values)
ppo_values_smooth = smooth_series(ppo_values)
pg_values_smooth = smooth_series(pg_values)

# Apply smoothing to daily returns
grpo_returns = compute_daily_returns(grpo_values)
ppo_returns = compute_daily_returns(ppo_values)
pg_returns = compute_daily_returns(pg_values)
grpo_returns_smooth = smooth_series(grpo_returns)
ppo_returns_smooth = smooth_series(ppo_returns)
pg_returns_smooth = smooth_series(pg_returns)

# Compute and smooth market baseline
market_values = compute_market_baseline(df_test, initial_amount=100000)
market_values_smooth = smooth_series(market_values)
market_returns = compute_daily_returns(market_values)
market_returns_smooth = smooth_series(market_returns)

dates = pd.to_datetime(df_test['date'].unique())
print(f"Length of dates: {len(dates)}")
print(f"Length of series: {len(grpo_values_smooth)}")

# Update the plot_series function
def plot_series(values_dict, title, ylabel, filename, bounds_dict=None, y_limits=None, dates=None):
    plt.figure(figsize=(10, 6))
    
    # Get the length of the first series
    first_series_len = len(next(iter(values_dict.values())))
    
    # Use indices if dates is None or lengths don't match
    if dates is None or len(dates) != first_series_len:
        x_values = range(first_series_len)
    else:
        x_values = dates
    
    for label, series in values_dict.items():
        plt.plot(x_values, series, label=label, linewidth=0.8)
        if bounds_dict and label in bounds_dict:
            min_series, max_series = bounds_dict[label]
            plt.fill_between(x_values, min_series, max_series, alpha=0.2)
    
    plt.xlabel("Date" if dates is not None else "Time Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if y_limits is not None:
        plt.ylim(y_limits)
    
    if dates is not None:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()

# Compute metrics for each strategy and market
metrics = {
    "GRPO": {
        "Sharpe Ratio": compute_sharpe_ratio(grpo_returns),
        "Sortino Ratio": compute_sortino_ratio(grpo_returns),
        "Calmar Ratio": compute_calmar_ratio(grpo_values),
        "Annualized Return": compute_annualized_return(grpo_values),
        "Annualized Volatility": compute_annualized_volatility(grpo_returns),
        "Max Drawdown": compute_max_drawdown(grpo_values),
        "Final Value": compute_final_value(grpo_values)
    },
    "PPO": {
        "Sharpe Ratio": compute_sharpe_ratio(ppo_returns),
        "Sortino Ratio": compute_sortino_ratio(ppo_returns),
        "Calmar Ratio": compute_calmar_ratio(ppo_values),
        "Annualized Return": compute_annualized_return(ppo_values),
        "Annualized Volatility": compute_annualized_volatility(ppo_returns),
        "Max Drawdown": compute_max_drawdown(ppo_values),
        "Final Value": compute_final_value(ppo_values)
    },
    "PG_EIIE": {
        "Sharpe Ratio": compute_sharpe_ratio(pg_returns),
        "Sortino Ratio": compute_sortino_ratio(pg_returns),
        "Calmar Ratio": compute_calmar_ratio(pg_values),
        "Annualized Return": compute_annualized_return(pg_values),
        "Annualized Volatility": compute_annualized_volatility(pg_returns),
        "Max Drawdown": compute_max_drawdown(pg_values),
        "Final Value": compute_final_value(pg_values)
    },
    "Market": {
        "Sharpe Ratio": compute_sharpe_ratio(market_returns),
        "Sortino Ratio": compute_sortino_ratio(market_returns),
        "Calmar Ratio": compute_calmar_ratio(market_values),
        "Annualized Return": compute_annualized_return(market_values),
        "Annualized Volatility": compute_annualized_volatility(market_returns),
        "Max Drawdown": compute_max_drawdown(market_values),
        "Final Value": compute_final_value(market_values)
    }
}

print("Performance Metrics:")
for name, vals in metrics.items():
    print(f"{name}:")
    for metric, value in vals.items():
        print(f"  {metric}: {value:.4f}")

# Compute bounds for each series
window = 10
# Print lengths for debugging
print("\nDiagnostic information:")
print(f"Dates length: {len(dates)}")
print(f"GRPO values length: {len(grpo_values_smooth)}")
print(f"PPO values length: {len(ppo_values_smooth)}")
print(f"PG_EIIE values length: {len(pg_values_smooth)}")
print(f"Market values length: {len(market_values_smooth)}")

# Ensure all series have the same length by trimming to the shortest one
min_length = min(
    len(dates),
    len(grpo_values_smooth),
    len(ppo_values_smooth),
    len(pg_values_smooth),
    len(market_values_smooth)
)

dates = dates[:min_length]
grpo_values_smooth = grpo_values_smooth[:min_length]
ppo_values_smooth = ppo_values_smooth[:min_length]
pg_values_smooth = pg_values_smooth[:min_length]
market_values_smooth = market_values_smooth[:min_length]

# Update bounds to match the new lengths
bounds_dict_value = {
    "GRPO": (
        grpo_values_smooth[:min_length],
        grpo_values_smooth[:min_length]
    ),
    "PPO": (
        ppo_values_smooth[:min_length],
        ppo_values_smooth[:min_length]
    ),
    "PG_EIIE": (
        pg_values_smooth[:min_length],
        pg_values_smooth[:min_length]
    ),
    "Market": (
        market_values_smooth[:min_length],
        market_values_smooth[:min_length]
    )
}

print("\nAfter trimming:")
print(f"All series length: {min_length}")


# Update the plotting calls to include dates
plot_series(
    {
        "GRPO": grpo_values_smooth,
        "PPO": ppo_values_smooth,
        "PG_EIIE": pg_values_smooth,
        "Market": market_values_smooth
    },
    title="Portfolio Value Over Time (Smoothed)",
    ylabel="Portfolio Value",
    filename="comparison_portfolio_value.png",
    bounds_dict=bounds_dict_value,
    dates=dates
)
print("All comparison plots saved in 'results/' directory.")
