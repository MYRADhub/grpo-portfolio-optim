# train_grpo.py
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

# Import FinRL modules
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler, data_split
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.main import check_and_make_directories

# ---------------------------
# Monkey-Patch DRLAgent.get_model to support "grpo"
# ---------------------------
from grpo import GRPOAgent  # import your custom GRPO agent

_original_get_model = DRLAgent.get_model

def get_model_with_grpo(self, model_name, device="cpu", model_kwargs=None, policy_kwargs=None):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if model_name.lower() == "grpo":
        lr = model_kwargs.get("lr", 0.001)
        gamma = model_kwargs.get("gamma", 0.99)
        group_size = model_kwargs.get("group_size", 4)
        epsilon = model_kwargs.get("epsilon", 0.15)
        beta = model_kwargs.get("beta", 0.0005)
        return GRPOAgent(self.env, policy_kwargs=policy_kwargs, lr=lr, gamma=gamma,
                         group_size=group_size, epsilon=epsilon, beta=beta, device=device)
    else:
        return _original_get_model(self, model_name, device, model_kwargs, policy_kwargs)

DRLAgent.get_model = get_model_with_grpo

# ---------------------------
# Set Up Directories
# ---------------------------
check_and_make_directories(["data", "trained_models", "tensorboard_log", "results"])

# ---------------------------
# Define Portfolio Tickers and Fetch Data
# ---------------------------
TOP_BRL = [
    "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA",
    "BBAS3.SA", "RENT3.SA", "LREN3.SA", "PRIO3.SA",
    "WEGE3.SA", "ABEV3.SA"
]
print(f"Number of tickers: {len(TOP_BRL)}")

# Download raw data (from 2011-01-01 to 2022-12-31)
df_raw = YahooDownloader(start_date='2011-01-01',
                         end_date='2022-12-31',
                         ticker_list=TOP_BRL).fetch_data()
print("Raw data shape:", df_raw.shape)

# ---------------------------
# Normalize Data
# ---------------------------
portfolio_norm_df = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(df_raw)
# Select required columns (must include "close")
df_portfolio = portfolio_norm_df[["date", "tic", "close", "high", "low"]]
print("Data shape after selecting required columns:", df_portfolio.shape)

# ---------------------------
# Split Data into Training and Testing Sets
# ---------------------------
df_train = data_split(df_portfolio, '2011-01-01', '2019-01-01')
df_test  = data_split(df_portfolio, '2020-01-01', '2020-12-31')
print("Training data shape:", df_train.shape)
print("Testing data shape:", df_test.shape)

# ---------------------------
# Create Portfolio Optimization Environments
# ---------------------------
env_kwargs = {
    "initial_amount": 100000,
    "comission_fee_pct": 0.0025,
    "time_window": 50,
    "features": ["close", "high", "low"],
    "normalize_df": None
}
env_train = PortfolioOptimizationEnv(df=df_train, **env_kwargs)
env_test  = PortfolioOptimizationEnv(df=df_test, **env_kwargs)

# ---------------------------
# Instantiate DRLAgent with training environment and get GRPO model
# ---------------------------
agent = DRLAgent(env_train)
model_kwargs = {
    "lr": 0.001,
    "gamma": 0.99,
    "group_size": 4,
    "epsilon": 0.15,
    "beta": 0.0005,
    "device": "cpu"
}
policy_kwargs = {"hidden_sizes": [128, 128]}  # Example policy parameters; adjust as needed.
grpo_model = agent.get_model("grpo", model_kwargs=model_kwargs, policy_kwargs=policy_kwargs)

# ---------------------------
# Train the GRPO Model
# ---------------------------
print("Starting GRPO training...")
grpo_model.train(total_timesteps=1000)  # Quick test training

# ---------------------------
# Evaluate the GRPO model on the test environment (recording actions)
# ---------------------------
actions_history = []
state = env_test.reset()
# Try to get date information from env_test, otherwise None
dates = env_test._date_memory if hasattr(env_test, "_date_memory") and env_test._date_memory is not None else None
done = False
while not done:
    action = grpo_model.predict(state)
    actions_history.append(action)
    state, reward, done, _ = env_test.step(action)
final_value = env_test._asset_memory["final"]
print("Final portfolio value (GRPO):", final_value)

# ---------------------------
# Plotting Functions
# ---------------------------
def plot_portfolio_value_history(values, filename="results/grpo_portfolio_value.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(values, label="Portfolio Value", color='blue', alpha=0.8)
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value")
    plt.title("GRPO Portfolio Value Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_rewards_from_values(values, filename="results/grpo_rewards.png"):
    rewards = [values[i] - values[i-1] for i in range(1, len(values))]
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward (Change in Value)", color='green', alpha=0.8)
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_actions(actions, filename="results/grpo_actions.png"):
    import numpy as np
    actions_arr = np.array(actions)  # shape: [timesteps, num_assets]
    plt.figure(figsize=(10, 6))
    for i in range(actions_arr.shape[1]):
        plt.plot(actions_arr[:, i], label=f"Asset {i}")
    plt.xlabel("Timestep")
    plt.ylabel("Action (Weight)")
    plt.title("Actions Taken Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Still produces an error
def generate_quantstats_report(portfolio_values, filename="results/quantstats_report.html"):
    import quantstats as qs
    import numpy as np
    # Convert portfolio_values to numeric if it's not already
    if isinstance(portfolio_values, (list, dict)):
        values = np.array([float(v) for v in portfolio_values.values()] if isinstance(portfolio_values, dict) else portfolio_values)
    else:
        values = portfolio_values
    # Create an index with daily frequency
    dt_index = pd.date_range(start=datetime.date.today(), periods=len(values), freq='D')
    price_series = pd.Series(data=values, index=dt_index)
    # Compute daily returns (percentage change)
    returns = price_series.pct_change().dropna()
    qs.reports.html(returns, output=filename, title="GRPO Portfolio Summary")

# ---------------------------
# Call Plot Functions
# ---------------------------
plot_portfolio_value_history(final_value, filename="results/grpo_portfolio_value.png")
plot_rewards_from_values(final_value, filename="results/grpo_rewards.png")
plot_actions(actions_history, filename="results/grpo_actions.png")
try:
    generate_quantstats_report(final_value, filename="results/quantstats_report.html")
except Exception as e:
    print(f"Error generating quantstats report: {e}")

print("Plots saved in the 'results' directory.")