import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from eval import evaluate_portfolio

from stable_baselines3 import PPO

# FinRL modules
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler, data_split
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.main import check_and_make_directories

# Set up directories
check_and_make_directories(["data", "trained_models", "tensorboard_log", "results"])

# Load and prepare data
TOP_WLRD = ["^GSPC", "^GDAXI", "^IXIC", "^RUT", "^N225"]

df_raw = YahooDownloader(start_date='1988-02-01', end_date='2024-04-30', ticker_list=TOP_WLRD).fetch_data()
df_raw = df_raw.pivot(index="date", columns="tic", values=["open", "high", "low", "close"])
df_raw = df_raw.ffill().dropna()
df_raw = df_raw.stack(level="tic").reset_index()

portfolio_norm_df = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(df_raw)
df_portfolio = portfolio_norm_df[["date", "tic", "close", "high", "low"]]

df_train = data_split(df_portfolio, '1988-02-01', '2015-12-31')
df_test  = data_split(df_portfolio, '2016-01-01', '2024-04-30')

# Create environments
env_kwargs = {
    "initial_amount": 100000,
    "comission_fee_pct": 0.0025,
    "time_window": 50,
    "features": ["close", "high", "low"],
    "normalize_df": None
}
env_train = PortfolioOptimizationEnv(df=df_train, **env_kwargs)
env_test  = PortfolioOptimizationEnv(df=df_test, **env_kwargs)

# Train PPO agent
print("Training PPO agent...")
from stable_baselines3.common.vec_env import DummyVecEnv
vec_env_train = DummyVecEnv([lambda: env_train])

ppo_model = PPO("MlpPolicy", vec_env_train, verbose=1, tensorboard_log="tensorboard_log/ppo", learning_rate=0.0003, n_steps=128, batch_size=64)
ppo_model.learn(total_timesteps=1000000)
ppo_model.save("trained_models/ppo_portfolio")

# -----------------------------
# Evaluate PPO agent
# -----------------------------
print("Evaluating PPO agent...")
state = env_test.reset()
actions_history = []
portfolio_values = [env_test._initial_amount]
done = False

while not done:
    action, _ = ppo_model.predict(state, deterministic=True)
    
    # Normalize action to ensure sum to 1
    action = np.clip(action, 0, 1)
    action /= np.sum(action) if np.sum(action) > 0 else 1

    actions_history.append(action)

    state, reward, done, _ = env_test.step(action)

    # Manually record current portfolio value
    
    current_value = env_test._portfolio_value
    portfolio_values.append(current_value)

final_value = portfolio_values[-1]
print("Final portfolio value (PPO):", final_value)

# -----------------------------
# Plotting
# -----------------------------
def plot_portfolio_value_history(values, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(values, label="Portfolio Value", color='orange')
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Value")
    plt.title("PPO Portfolio Value Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_rewards_from_values(values, filename):
    rewards = [values[i] - values[i-1] for i in range(1, len(values))]
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Reward", color='purple')
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_actions(actions, filename):
    actions_arr = np.array(actions)
    plt.figure(figsize=(10, 6))
    for i in range(actions_arr.shape[1]):
        plt.plot(actions_arr[:, i], label=f"Asset {i}")
    plt.xlabel("Timestep")
    plt.ylabel("Action (Weight)")
    plt.title("PPO Actions Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_daily_returns(values, filename):
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

def plot_drawdown(values, filename):
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

# Save plots
plot_portfolio_value_history(portfolio_values, "results/ppo_portfolio_value.png")
plot_rewards_from_values(portfolio_values, "results/ppo_rewards.png")
plot_actions(actions_history, "results/ppo_actions.png")
plot_daily_returns(portfolio_values, "results/ppo_daily_returns.png")
plot_drawdown(portfolio_values, "results/ppo_drawdown.png")

# Evaluate metrics
evaluate_portfolio(portfolio_values, model_name="PPO")
