"""
EconNet Demo -- Compare all four forecasting models on synthetic tasks.

This script demonstrates the full EconNet workflow:
1. Generate synthetic AR(2) and economic data
2. Train each model (LSTM, TCN, Transformer, N-BEATS)
3. Evaluate and compare performance (MSE, MAE, parameter counts)
"""

import sys
import os

# Allow running from project root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from econnet.models import (
    LSTMForecaster,
    TCNForecaster,
    TransformerForecaster,
    NBEATSForecaster,
)
from econnet.datasets import generate_arma, generate_economic_data
from econnet.utils import train_test_split, StandardScaler, mse, mae, count_parameters


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Lightweight settings for quick demo
EPOCHS = 30       # training epochs per model
SEQ_LEN = 10      # input sequence length
TEST_SIZE = 0.2   # fraction reserved for testing

MODELS = {
    "LSTM": LSTMForecaster(hidden_size=16, num_layers=2, dropout=0.05, random_state=42),
    "TCN": TCNForecaster(num_channels=[16, 16], kernel_size=3, dropout=0.05, random_state=42),
    "Transformer": TransformerForecaster(
        d_model=16, n_heads=4, n_layers=2, d_ff=32, dropout=0.05, random_state=42
    ),
    "N-BEATS": NBEATSForecaster(
        n_blocks=2, n_layers=3, hidden_size=32, theta_dim=32,
        trend_degree=2, n_harmonics=3, period=7.0, random_state=42,
    ),
}

BANNER_WIDTH = 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_banner(title: str) -> None:
    """Print a centred banner."""
    print()
    print("=" * BANNER_WIDTH)
    print(f"  {title}")
    print("=" * BANNER_WIDTH)


def print_model_header(name: str) -> None:
    """Print a model section header."""
    print(f"\n{'─' * BANNER_WIDTH}")
    print(f"  [Training] {name}")
    print(f"{'─' * BANNER_WIDTH}")


def evaluate_model(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train, predict, and evaluate a single model.

    Returns a dict with keys: name, mse, mae, params, train_time.
    """
    import time

    print_model_header(name)

    t0 = time.time()
    model.fit(X_train, y_train, epochs=EPOCHS, lr=0.001, verbose=True)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    mse_val = mse(y_test, y_pred)
    mae_val = mae(y_test, y_pred)
    n_params = model.count_params()

    print(f"  MSE = {mse_val:.6f}  |  MAE = {mae_val:.6f}  |  Params = {n_params}")

    return {
        "name": name,
        "mse": mse_val,
        "mae": mae_val,
        "params": n_params,
        "time": train_time,
    }


def print_results_table(results: list, task_name: str) -> None:
    """Print a formatted results table."""
    print(f"\n{'─' * BANNER_WIDTH}")
    print(f"  Results: {task_name}")
    print(f"{'─' * BANNER_WIDTH}")
    header = f"  {'Model':<16s} {'MSE':>10s} {'MAE':>10s} {'Params':>8s} {'Time(s)':>8s}"
    print(header)
    print(f"  {'─' * 14}  {'─' * 8}  {'─' * 8}  {'─' * 6}  {'─' * 6}")

    for r in results:
        print(
            f"  {r['name']:<16s} "
            f"{r['mse']:10.6f} "
            f"{r['mae']:10.6f} "
            f"{r['params']:8d} "
            f"{r['time']:8.2f}"
        )


# ---------------------------------------------------------------------------
# Task 1: AR(2) Forecasting
# ---------------------------------------------------------------------------

def run_ar2_task() -> None:
    """Generate AR(2) data and compare all models."""
    print_banner("Task 1: AR(2) Process Forecasting")

    # Generate AR(2): y_t = 0.6*y_{t-1} - 0.3*y_{t-2} + epsilon
    np.random.seed(123)
    y_series, _ = generate_arma(n=1200, ar=(0.6, -0.3), ma=(0.0,), noise_std=0.1, random_state=123)

    print(f"  AR(2) series generated: {len(y_series)} points")
    print(f"  Mean = {np.mean(y_series):.4f}, Std = {np.std(y_series):.4f}")

    # Build sequences
    from econnet.utils import create_sequences
    X, y = create_sequences(y_series, seq_len=SEQ_LEN, horizon=1)
    print(f"  Input shape: {X.shape}, Target shape: {y.shape}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    # Normalize (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print(f"  Train: {X_train_s.shape[0]} samples, Test: {X_test_s.shape[0]} samples")

    results = []
    for name, model in MODELS.items():
        # N-BEATS needs special shape handling (it flattens internally)
        r = evaluate_model(name, model, X_train_s, y_train, X_test_s, y_test)
        results.append(r)

    print_results_table(results, "AR(2) Forecasting")


# ---------------------------------------------------------------------------
# Task 2: Economic Data Forecasting
# ---------------------------------------------------------------------------

def run_economic_task() -> None:
    """Generate synthetic economic data and compare all models."""
    print_banner("Task 2: Economic Data Forecasting (GDP)")

    np.random.seed(456)
    X, y = generate_economic_data(n=800, random_state=456)

    print(f"  Features: GDP, Inflation, Unemployment")
    print(f"  Input shape: {X.shape}, Target shape: {y.shape}")
    print(f"  Target (GDP) mean = {np.mean(y):.4f}, std = {np.std(y):.4f}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print(f"  Train: {X_train_s.shape[0]} samples, Test: {X_test_s.shape[0]} samples")

    results = []
    for name, model in MODELS.items():
        r = evaluate_model(name, model, X_train_s, y_train, X_test_s, y_test)
        results.append(r)

    print_results_table(results, "Economic Data (GDP Forecasting)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full EconNet demonstration."""
    print_banner("EconNet Demo: Deep Learning for Economic Time Series Forecasting")
    print(f"\n  Models: {', '.join(MODELS.keys())}")
    print(f"  Epochs per task: {EPOCHS}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Test split:      {TEST_SIZE}")

    # ---- Task 1 ----
    run_ar2_task()

    # ---- Task 2 ----
    run_economic_task()

    # ---- Summary ----
    print_banner("Demo Complete")
    print("\n  All models have been trained and evaluated.")
    print("  Check the tables above for MSE, MAE, and parameter counts.")
    print("  All implementations are pure NumPy -- zero framework dependency.\n")


if __name__ == "__main__":
    main()
