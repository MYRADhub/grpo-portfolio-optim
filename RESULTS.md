# Benchmark results: GRPO vs PPO vs PG/EIIE vs buy-and-hold

> These numbers were measured by a portfolio audit run of this repository, not by the
> repository's authors. The repo reports no metric values anywhere; `results/` and
> `trained_models/` are gitignored. Everything below was produced by running the repo's
> own `train_grpo.py` / `train_ppo.py` / `train_pg.py` code paths and `plots.py`.

## Experimental setup

| | |
|---|---|
| Assets | `^GSPC` (S&P 500), `^GDAXI` (DAX), `^IXIC` (NASDAQ), `^RUT` (Russell 2000), `^N225` (Nikkei 225), plus cash — a 6-dimensional simplex action |
| Data | Yahoo Finance daily OHLCV, 1988-02-01 to 2024-04-29, forward-filled and inner-joined across tickers; per-ticker MaxAbs normalisation (`GroupByScaler`) |
| Environment | FinRL `PortfolioOptimizationEnv`, features `close/high/low`, 50-day observation window, 0.25% commission on rebalances, $100,000 initial capital |
| Train split | **1988-02-01 to 2015-12-30**, 7,264 trading days |
| Test split (out-of-sample) | **2016-01-04 to 2024-04-29**, 2,166 trading days → 2,116 evaluated steps after the 50-day window and series alignment |
| Timestep budget | GRPO **1,000,000 timesteps**, PPO **1,000,000 timesteps** (both the value hardcoded in the repo's training scripts), PG/EIIE **100 episodes** (the repo's value; ≈726,400 environment steps) |
| Seeds | **3 seeds (0, 1, 2)**, reported as mean ± std across seeds |
| Train wall time (per seed, CPU) | GRPO ≈ 2,070 s, PPO ≈ 2,142 s, EIIE ≈ 1,474 s; all nine runs executed in parallel, ≈36 min total |
| Metric definitions | Taken verbatim from the repo's `plots.py`: daily returns are `pct_change().fillna(0)`, annualised return is geometric (CAGR at 252 trading days), Sharpe and Sortino use a zero risk-free rate, Calmar is CAGR / abs(max drawdown) |
| Baseline | S&P 500 buy-and-hold over the identical test window, as computed by `plots.py`'s `compute_market_baseline` |

## Results, out-of-sample 2016-01-04 to 2024-04-29 (mean ± std over 3 seeds)

| Strategy | Cumulative Return | Annualised Return | Annualised Volatility | Sharpe | Sortino | Calmar | Max Drawdown | Final Value |
|---|---|---|---|---|---|---|---|---|
| GRPO (this repo, from scratch) | 114.1% ± 14.6 | 9.5% ± 0.9 | 13.4% ± 0.8 | 0.741 ± 0.058 | 0.929 ± 0.090 | 0.323 ± 0.025 | -29.3% ± 0.9 | $214,086 ± 14,600 |
| PPO (Stable-Baselines3) | 91.0% ± 20.0 | 8.0% ± 1.3 | 12.1% ± 2.2 | 0.694 ± 0.053 | 0.860 ± 0.050 | 0.278 ± 0.019 | -28.6% ± 4.4 | $190,977 ± 19,962 |
| PG / EIIE (FinRL) | 122.9% ± 3.0 | 10.0% ± 0.2 | 13.9% ± 0.6 | 0.756 ± 0.017 | 0.918 ± 0.023 | 0.335 ± 0.010 | -29.9% ± 1.4 | $222,923 ± 3,039 |
| S&P 500 buy & hold | 153.3% | 11.7% | 18.1% | 0.704 | 0.824 | 0.345 | -33.9% | $254,199 |

Per-seed detail (the spread matters — see the caveats):

| Strategy | Seed | Cumulative Return | Sharpe | Sortino | Calmar | Max Drawdown |
|---|---|---|---|---|---|---|
| GRPO | 0 | 112.5% | 0.790 | 0.969 | 0.332 | -28.3% |
| GRPO | 1 | 97.0% | 0.659 | 0.804 | 0.289 | -29.1% |
| GRPO | 2 | 132.7% | 0.773 | 1.013 | 0.348 | -30.4% |
| PPO | 0 | 68.9% | 0.634 | 0.818 | 0.254 | -25.4% |
| PPO | 1 | 86.7% | 0.763 | 0.931 | 0.302 | -25.6% |
| PPO | 2 | 117.3% | 0.685 | 0.832 | 0.279 | -34.7% |
| PG / EIIE | 0 | 127.1% | 0.733 | 0.885 | 0.322 | -31.8% |
| PG / EIIE | 1 | 119.9% | 0.762 | 0.927 | 0.339 | -29.0% |
| PG / EIIE | 2 | 121.8% | 0.773 | 0.941 | 0.345 | -28.8% |

## What the numbers say

- **No agent beat the market on total return.** Over 2016-2024 the S&P 500 returned 153.3%
  versus 122.9% (EIIE), 114.1% (GRPO) and 91.0% (PPO). Buy-and-hold also has the best Calmar
  ratio (0.345).
- **GRPO and EIIE beat the market on risk-adjusted return**, because they run at roughly
  three quarters of the market's volatility (13.4% and 13.9% versus 18.1%) and take a
  shallower worst drawdown (-29% versus -33.9%): Sharpe 0.741 and 0.756 versus 0.704,
  Sortino 0.929 and 0.918 versus 0.824. PPO does not (Sharpe 0.694).
- **GRPO beats PPO on every metric reported here** at the same 1,000,000-timestep budget —
  the comparison the project set out to make.
- **GRPO does not beat the EIIE policy-gradient baseline.** EIIE is ahead on cumulative
  return, annualised return and Calmar; the Sharpe gap (0.741 vs 0.756) is well inside the
  seed-to-seed spread. EIIE is also far more stable across seeds (cumulative return spread
  119.9-127.1%, versus 97.0-132.7% for GRPO).
- **The volatility reduction, not stock-picking, is doing the work.** Averaged over the test
  episode the applied portfolio weights are close to equal weight for two of three GRPO
  seeds (seed 0: 0.096-0.280 across the six positions; seed 1: 0.090-0.259; equal weight is
  0.167). Seed 2 is the exception, parking 63.5% in the Nikkei. The weights also barely move
  through time — the per-position standard deviation across all 2,116 test days is 0.003-0.028
  for seeds 0 and 1 — so the learned policy is effectively a fixed allocation rather than a
  state-dependent timing policy.

## Caveats, stated plainly

- **Three seeds is thin.** The GRPO cumulative-return spread across seeds is 35 percentage
  points and the PPO spread is 48. Differences between GRPO and EIIE of the size seen here
  are not statistically established by three runs.
- **One train/test split, one asset universe, no hyperparameter search.** Every hyperparameter
  is the repo's committed value.
- These are audit numbers reproduced from the repo's code, on data re-downloaded from Yahoo
  Finance in 2026. They are not the authors' reported results, because the repo reports none.
- The out-of-sample window (2016-2024) contains one of the strongest equity bull markets on
  record, which flatters buy-and-hold and penalises any agent holding cash.

## Reproducing

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install numpy pandas scikit-learn matplotlib 'gym==0.26.2' gymnasium \
               stable-baselines3 quantstats yfinance tqdm stockstats torch-geometric
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
python train_grpo.py && python train_ppo.py && python train_pg.py && python plots.py
```

Two environment notes that are not repo defects: FinRL's package `__init__` eagerly imports
its Alpaca and RLlib entry points and must be bypassed (import the
`finrl.meta.env_portfolio_optimization` / `finrl.agents.portfolio_optimization` subtree
directly), and FinRL's `YahooDownloader` passes a `proxy=` argument that current yfinance no
longer accepts, so prices were pulled once with yfinance directly and cached to CSV — every
agent above sees byte-identical prices.
