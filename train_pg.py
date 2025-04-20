import logging
import datetime
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from eval import evaluate_portfolio

# Disable matplotlib font warnings
logging.getLogger('matplotlib.font_manager').disabled = True

# ---------------------------
# Import FinRL Modules
# ---------------------------
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler, data_split
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.main import check_and_make_directories

# ---------------------------
# Create Required Directories
# ---------------------------
check_and_make_directories(["data", "trained_models", "tensorboard_log", "results"])

# ---------------------------
# Define tickers and Fetch Data
# ---------------------------
TOP_WLRD = [
    "^GSPC",   # S&P 500 🇺🇸
    "^GDAXI",  # DAX 30 🇩🇪
    "^IXIC",   # NASDAQ Composite 🇺🇸
    "^RUT",    # Russell 2000 🇺🇸
    "^N225",   # Nikkei 225 🇯🇵
]
print(f"Number of tickers: {len(TOP_WLRD)}")

portfolio_raw_df = YahooDownloader(start_date='1988-02-01',
                                   end_date='2024-04-30',
                                   ticker_list=TOP_WLRD).fetch_data()
print("Raw data shape:", portfolio_raw_df.shape)

# Handle missing values
portfolio_raw_df = portfolio_raw_df.pivot(index="date", columns="tic", values=["open", "high", "low", "close", "volume"])
portfolio_raw_df = portfolio_raw_df.fillna(method="ffill").dropna()
portfolio_raw_df = portfolio_raw_df.stack(level="tic").reset_index()

# Normalize
portfolio_norm_df = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(portfolio_raw_df)
df_portfolio = portfolio_norm_df[["date", "tic", "close", "high", "low"]]

# ---------------------------
# Train-Test Split (like GRPO)
# ---------------------------
df_train = data_split(df_portfolio, '1988-02-01', '2015-12-31')
df_test  = data_split(df_portfolio, '2016-01-01', '2024-04-30')
print("Training data shape:", df_train.shape)
print("Testing data shape:", df_test.shape)

# ---------------------------
# Environment Setup
# ---------------------------
env_kwargs = {
    "initial_amount": 100000,
    "comission_fee_pct": 0.0025,
    "time_window": 50,
    "features": ["close", "high", "low"],
    "normalize_df": None
}
environment_train = PortfolioOptimizationEnv(df=df_train, **env_kwargs)
environment_test  = PortfolioOptimizationEnv(df=df_test, **env_kwargs)

# ---------------------------
# Train PG Agent with EIIE
# ---------------------------
agent = DRLAgent(environment_train)
model_kwargs = {
    "lr": 0.0003,
    "policy": EIIE
}
policy_kwargs = {
    "time_window": env_kwargs["time_window"],
    "k_size": 3
}
model = agent.get_model("pg", model_kwargs=model_kwargs, policy_kwargs=policy_kwargs)
trained_model = DRLAgent.train_model(model, episodes=5)

# ---------------------------
# Evaluate on Test Environment
# ---------------------------
# This automatically populates ._asset_memory and ._action_memory
DRLAgent.DRL_validation(trained_model, environment_test)
final_asset_values = environment_test._asset_memory["final"]
actions_history = environment_test._action_memory if hasattr(environment_test, "_action_memory") else None
print("Final asset value from test environment:", final_asset_values)

# ---------------------------
# Save the Trained Model
# ---------------------------
save_path = "trained_models/pg_eiie.pt"
torch.save(trained_model.train_policy.state_dict(), save_path)
print(f"Trained model saved to {save_path}")

# ---------------------------
# Plotting Functions
# ---------------------------
def plot_portfolio_value_history(values, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(values, label="Portfolio Value (PG/EIIE)", color='blue')
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_rewards_from_values(values, filename):
    rewards = [values[i] - values[i - 1] for i in range(1, len(values))]
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward", color='green')
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_actions(actions, filename):
    if actions is None:
        print("No action memory found.")
        return
    actions_arr = np.array(actions)
    plt.figure(figsize=(10, 6))
    for i in range(actions_arr.shape[1]):
        plt.plot(actions_arr[:, i], label=f"Asset {i}")
    plt.xlabel("Timestep")
    plt.ylabel("Weight")
    plt.title("Action Weights Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_daily_returns(values, filename="results/pg_daily_returns.png"):
    returns = pd.Series(values).pct_change().fillna(0)
    plt.figure(figsize=(10, 6))
    plt.plot(returns, label="Daily Returns", color='steelblue')
    plt.xlabel("Timestep")
    plt.ylabel("Return")
    plt.title("Daily Returns Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_drawdown(values, filename="results/pg_drawdown.png"):
    cumulative = pd.Series(values)
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    plt.figure(figsize=(10, 6))
    plt.plot(drawdown, label="Drawdown", color='crimson')
    plt.xlabel("Timestep")
    plt.ylabel("Drawdown")
    plt.title("Drawdown Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ---------------------------
# Generate and Save Plots
# ---------------------------
plot_portfolio_value_history(final_asset_values, filename="results/pg_portfolio_value.png")
plot_rewards_from_values(final_asset_values, filename="results/pg_rewards.png")
plot_actions(actions_history, filename="results/pg_actions.png")
plot_daily_returns(final_asset_values, filename="results/pg_daily_returns.png")
plot_drawdown(final_asset_values, filename="results/pg_drawdown.png")

# ---------------------------
# Evaluate the Results
# ---------------------------
evaluate_portfolio(final_asset_values, model_name="PG_EIIE")
