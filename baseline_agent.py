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
from finrl.plot import backtest_stats, backtest_plot, get_baseline

# ---------------------------
# 2. Create Required Directories
# ---------------------------
check_and_make_directories(["data", "trained_models", "tensorboard_log", "results"])

# ---------------------------
# 3. Define tickers and Fetch Data
# ---------------------------
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
# We'll work only with the columns needed by the environment.
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
env_kwargs = {
    "initial_amount": 100000,         # Starting cash amount
    "comission_fee_pct": 0.0025,        # Transaction cost percentage
    "time_window": 50,                # Time window (number of timesteps per episode)
    "features": ["close", "high", "low"],  # Features used by the environment
    "normalize_df": None              # Data is already normalized
}
environment_train = PortfolioOptimizationEnv(df=df_train, **env_kwargs)
environment_test  = PortfolioOptimizationEnv(df=df_test, **env_kwargs)

# ---------------------------
# 7. Initialize and Train a PG (Policy Gradient) Model Using the EIIE Architecture
# ---------------------------
agent = DRLAgent(environment_train)

# Set PG algorithm parameters and specify the EIIE architecture via the 'policy' field.
model_kwargs = {
    "lr": 0.01,    # Learning rate for the policy
    "policy": EIIE # Use the EIIE policy architecture
}
policy_kwargs = {
    "time_window": env_kwargs["time_window"],
    "k_size": 3   # A parameter of the EIIE architecture (kernel size)
}

# Get the model (using "pg" since FinRL's portfolio optimization currently supports PG).
model = agent.get_model("pg", model_kwargs=model_kwargs, policy_kwargs=policy_kwargs)

# Train the model for a few episodes (here, 5 episodes for a quick test)
trained_model = DRLAgent.train_model(model, episodes=5)

# ---------------------------
# 8. Validate the Trained Model on the Test Environment
# ---------------------------
# This will run the model on the testing environment using DRLAgent's built-in validation.
DRLAgent.DRL_validation(trained_model, environment_test)

# Retrieve the portfolio value history from the test environment.
final_asset_values = environment_test._asset_memory["final"]
print("Final asset value from test environment:", final_asset_values)

# ---------------------------
# 9. Save the Trained Model
# ---------------------------
# Here we save the state dictionary of the trained policy network.
# Depending on the DRL model used, the policy network is typically stored in an attribute.
# In our PG agent (based on EIIE), it's available as `train_policy`.
save_path = "trained_models/trained_policy_EIIE.pt"
torch.save(trained_model.train_policy.state_dict(), save_path)
print(f"Trained model saved to {save_path}")

# ---------------------------
# 10. Quick Plot to Verify Results
# ---------------------------
plt.figure(figsize=(8, 4))
plt.plot(final_asset_values, label="Portfolio Value (PG/EIIE)")
plt.xlabel("Timestep")
plt.ylabel("Portfolio Value")
plt.title("Portfolio Optimization Test with PG (EIIE)")
plt.legend()
plt.tight_layout()
plt.show()
