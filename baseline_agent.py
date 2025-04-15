import logging
import datetime
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

# Disable matplotlib font warnings
logging.getLogger('matplotlib.font_manager').disabled = True

# ---------------------------
# 1. Import FinRL Modules and Utilities
# ---------------------------
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE
from finrl.main import check_and_make_directories
# from finrl.config_tickers import TOP_BRL  # If not defined, we will override it below.
from finrl.plot import backtest_stats, backtest_plot, get_baseline

# ---------------------------
# 2. Create Required Directories
# ---------------------------
check_and_make_directories(["data", "trained_models", "tensorboard_log", "results"])

# ---------------------------
# 3. Define tickers and Fetch Data
# ---------------------------
# Define a list for 10 Brazilian stocks (this list is similar to the sample)
TOP_BRL = [
    "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA",
    "BBAS3.SA", "RENT3.SA", "LREN3.SA", "PRIO3.SA",
    "WEGE3.SA", "ABEV3.SA"
]
print(f"Number of tickers: {len(TOP_BRL)}")

# Download raw data (from 2011 to 2022)
portfolio_raw_df = YahooDownloader(start_date='2011-01-01',
                                   end_date='2022-12-31',
                                   ticker_list=TOP_BRL).fetch_data()
print("Raw data shape:", portfolio_raw_df.shape)

# ---------------------------
# 4. Normalize Data
# ---------------------------
# Use GroupByScaler with MaxAbsScaler to scale each stock’s series into [0, 1]
portfolio_norm_df = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(portfolio_raw_df)
# We'll work only with a few columns needed by the environment.
df_portfolio = portfolio_norm_df[["date", "tic", "close", "high", "low"]]

# ---------------------------
# 5. Split Data into Training and Test Sets
# ---------------------------
df_train = df_portfolio[(df_portfolio["date"] >= "2011-01-01") & (df_portfolio["date"] < "2019-01-01")]
df_test  = df_portfolio[(df_portfolio["date"] >= "2020-01-01") & (df_portfolio["date"] < "2020-12-31")]
print("Training data shape:", df_train.shape)
print("Test data shape:", df_test.shape)

# ---------------------------
# 6. Instantiate the Portfolio Optimization Environment
# ---------------------------
# For portfolio optimization, we set a time window (e.g., 50 days) and pass the features to be used.
env_kwargs = {
    "initial_amount": 100000,         # starting cash
    "comission_fee_pct": 0.0025,        # commission fee (2.5 basis points)
    "time_window": 50,                # time window for each episode
    "features": ["close", "high", "low"],  # features used by the environment
    "normalize_df": None              # Data is already normalized
}
environment_train = PortfolioOptimizationEnv(df=df_train, **env_kwargs)
environment_test  = PortfolioOptimizationEnv(df=df_test, **env_kwargs)

# ---------------------------
# 7. Initialize and Train a PG (Policy Gradient) Model Using the EIIE Architecture
# ---------------------------
# In FinRL's portfolio optimization module, the available algorithm is "pg" (Policy Gradient).
agent = DRLAgent(environment_train)

# Set PG algorithm parameters and the EIIE architecture's parameters.
model_kwargs = {
    "lr": 0.01,    # learning rate
    "policy": EIIE # use the EIIE architecture
}
policy_kwargs = {
    "time_window": env_kwargs["time_window"],
    "k_size": 3
}

# Get the model using "pg" (note: do not use "ppo" here)
model = agent.get_model("pg", model_kwargs=model_kwargs, policy_kwargs=policy_kwargs)

# Train the model on the training environment for a few episodes (here, 5 for a quick test)
trained_model = DRLAgent.train_model(model, episodes=5)

# ---------------------------
# 8. Validate the Trained Model on the Test Environment
# ---------------------------
# Use the DRLAgent.DRL_validation method to run the trained policy on the test environment.
DRLAgent.DRL_validation(trained_model, environment_test)

# Retrieve the portfolio value history from the test environment.
final_asset_values = environment_test._asset_memory["final"]
print("Final asset value from test environment:", final_asset_values)

# ---------------------------
# 9. Quick Plot to Verify Results
# ---------------------------
plt.figure(figsize=(8, 4))
plt.plot(final_asset_values, label="Portfolio Value (PG/EIIE)")
plt.xlabel("Timestep")
plt.ylabel("Portfolio Value")
plt.title("Portfolio Optimization Test with PG (EIIE)")
plt.legend()
plt.tight_layout()
plt.show()
