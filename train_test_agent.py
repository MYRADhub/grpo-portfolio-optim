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
# Monkey-Patch DRLAgent.get_model to support the "test" model.
# ---------------------------
from test_agent import TestAgentModel  # import your external test model

_original_get_model = DRLAgent.get_model

def get_model_test(self, model_name, device="cpu", model_kwargs=None, policy_kwargs=None):
    model_kwargs = {} if model_kwargs is None else model_kwargs
    policy_kwargs = {} if policy_kwargs is None else policy_kwargs
    if model_name.lower() == "test":
        return TestAgentModel(self.env, device=device, **model_kwargs)
    else:
        return _original_get_model(self, model_name, device, model_kwargs, policy_kwargs)

DRLAgent.get_model = get_model_test

# ---------------------------
# Set Up Directories
# ---------------------------
check_and_make_directories(["data", "trained_models", "tensorboard_log", "results"])

# ---------------------------
# Define Portfolio Tickers and Fetch Data
# ---------------------------
# Using 10 Brazilian stocks as an example:
TOP_BRL = [
    "VALE3.SA", "PETR4.SA", "ITUB4.SA", "BBDC4.SA",
    "BBAS3.SA", "RENT3.SA", "LREN3.SA", "PRIO3.SA",
    "WEGE3.SA", "ABEV3.SA"
]
print(f"Number of tickers: {len(TOP_BRL)}")

# Download raw data from Yahoo Finance (from 2011 to 2022)
df_raw = YahooDownloader(start_date='2011-01-01',
                         end_date='2022-12-31',
                         ticker_list=TOP_BRL).fetch_data()
print("Raw data shape:", df_raw.shape)

# ---------------------------
# Normalize Data
# ---------------------------
# Use GroupByScaler with MaxAbsScaler to scale each ticker's data.
portfolio_norm_df = GroupByScaler(by="tic", scaler=MaxAbsScaler).fit_transform(df_raw)
# Extract only the columns needed by the environment.
df_portfolio = portfolio_norm_df[["date", "tic", "close", "high", "low"]]
print("Shape of DataFrame after selecting columns:", df_portfolio.shape)

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
# Instantiate DRLAgent with the Training Environment and Get TestAgent Model
# ---------------------------
agent = DRLAgent(env_train)
# Use "test" as model_name to get your custom TestAgentModel.
test_model = agent.get_model("test", model_kwargs={})

# ---------------------------
# Train the TestAgent Model
# ---------------------------
print("Training TestAgentModel (should display random performance)...")
test_model.train(episodes=5)

# ---------------------------
# Evaluate the TestAgent Model on the Test Environment
# ---------------------------
print("Starting test rollout on the test environment...")
state = env_test.reset()
done = False
while not done:
    action = test_model.predict(state)
    state, reward, done, _ = env_test.step(action)
final_value = env_test._asset_memory["final"]
print("Final portfolio value (TestAgent):", final_value)

# ---------------------------
# Plot the Portfolio Value History
# ---------------------------
plt.figure(figsize=(8, 4))
plt.plot(env_test._asset_memory["final"], label="TestAgent Portfolio Value")
plt.xlabel("Timestep")
plt.ylabel("Portfolio Value")
plt.title("Test Agent Portfolio Optimization")
plt.legend()
plt.tight_layout()
plt.show()
