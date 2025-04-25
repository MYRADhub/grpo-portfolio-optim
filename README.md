# GRPO for Portfolio Management

This repository implements and benchmarks **Grouped Regularized Policy Optimization (GRPO)** for financial portfolio optimization using historical market data. We compare GRPO against standard baselines including **Proximal Policy Optimization (PPO)** and a **Policy Gradient agent with EIIE architecture**, using the [FinRL](https://github.com/AI4Finance-Foundation/FinRL) environment.

---

## Overview

- **GRPO**: A PPO-style algorithm that removes the value network and uses group-based advantage estimation with KL-regularized updates.
- **PPO**: Standard actor-critic algorithm implemented via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
- **PG (EIIE)**: A convolutional policy gradient model designed for time-series portfolio data. (Built into FinRL)

The agents are evaluated on a multi-index dataset containing daily data from S&P 500, NASDAQ, DAX, Russell 2000, and Nikkei 225.

---

## Project Structure

```
.
├── train_grpo.py        # Training script for GRPO
├── train_ppo.py         # Training script for PPO (Stable-Baselines3)
├── train_pg.py          # Training script for PG with EIIE
├── grpo.py              # GRPO agent implementation
├── eval.py              # Evaluation and metrics utilities
├── plots.py             # Generates plots and computes final metrics
├── requirements.txt     # List of Python dependencies
├── trained_models/      # (Auto-generated) Saved model weights
└── results/             # (Auto-generated) Plot outputs
```

---

## Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/MYRADhub/grpo-portfolio-optim.git
cd grpo-portfolio-optim
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Tested with Python 3.10

---

## Training Agents

Run each script to train models:

```bash
python train_grpo.py  # Train GRPO
python train_ppo.py   # Train PPO (Stable-Baselines3)
python train_pg.py    # Train PG (EIIE)
```

Models will be saved in the `trained_models/` folder.

---

## Evaluation & Visualization

Once models are trained, run:

```bash
python plots.py
```

This will:
- Evaluate all agents on the 2016–2024 test set
- Generate plots and metrics in the `results/` directory
- Print Sharpe, Sortino, Calmar, and other metrics

You can also use `eval.py` to access standalone evaluation utilities.

---

## Metrics Reported

- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Annualized Return
- Volatility
- Maximum Drawdown
- Final Portfolio Value

---

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Includes:
- `finrl`
- `stable-baselines3`
- `torch`
- `scikit-learn`
- `matplotlib`
- `pandas`, `numpy`, etc.

---

## References

- [GRPO (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [FinRL Library](https://arxiv.org/abs/1706.10059)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [EIIE Architecture](https://sol.sbc.org.br/index.php/bwaif/article/view/24959)

---

## Authors

- **Murad Ismayilov** – [murad.ismayilov@mail.mcgill.ca](mailto:murad.ismayilov@mail.mcgill.ca)  
- **Jalil Jabbarli** – [jalil.jabbarli@mail.mcgill.ca](mailto:jalil.jabbarli@mail.mcgill.ca)  
- McGill University

---

## License

This project has MIT open-access licence.