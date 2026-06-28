"""
Synthetic economic data generators for EconNet.

Provides functions to generate realistic synthetic economic and financial
time series for testing and benchmarking forecasting models.
"""

import numpy as np
from typing import Optional, Tuple


def generate_arma(
    n: int = 1000,
    ar: Tuple[float, ...] = (0.6, -0.3),
    ma: Tuple[float, ...] = (0.2,),
    noise_std: float = 0.1,
    burn_in: int = 200,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an ARMA(p, q) process.

    The process is defined as:

        y_t = c + sum_{i=1}^p ar_i * y_{t-i} + sum_{j=1}^q ma_j * eps_{t-j} + eps_t

    where eps_t ~ N(0, noise_std^2).

    Parameters
    ----------
    n : int
        Number of observations to generate (after burn-in).
    ar : tuple of float
        Autoregressive coefficients (AR).
    ma : tuple of float
        Moving-average coefficients (MA).
    noise_std : float
        Standard deviation of the white noise innovations.
    burn_in : int
        Number of initial time steps to discard.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    y : np.ndarray, shape (n,)
        The generated ARMA time series.
    eps : np.ndarray, shape (n,)
        The white noise innovations.
    """
    if random_state is not None:
        np.random.seed(random_state)

    p = len(ar)
    q = len(ma)
    total_len = n + burn_in

    eps = np.random.randn(total_len) * noise_std
    y = np.zeros(total_len)

    for t in range(max(p, q), total_len):
        ar_term = sum(ar[i] * y[t - i - 1] for i in range(p))
        ma_term = sum(ma[j] * eps[t - j - 1] for j in range(q))
        y[t] = ar_term + ma_term + eps[t]

    return y[burn_in:], eps[burn_in:]


def generate_economic_data(
    n: int = 1000,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic economic data with realistic cross-correlations.

    Simulates three macroeconomic variables:

    1. **GDP** (output gap) -- persistent cycle around a long-run trend
    2. **Inflation** (CPI) -- positively correlated with GDP (Phillips curve)
    3. **Unemployment** -- negatively correlated with GDP (Okun's law)

    The target ``y`` is the next-period GDP value.

    Parameters
    ----------
    n : int
        Number of time steps to generate.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n - seq_len, seq_len, 3)
        Input sequences of length 10 with 3 features (gdp, inflation, unemployment).
    y : np.ndarray, shape (n - seq_len,)
        Target values (next-step GDP).
    """
    if random_state is not None:
        np.random.seed(random_state)

    seq_len = 10

    # -- GDP: AR(2) with moderate persistence ---------------------------------
    gdp = np.zeros(n + 2)
    for t in range(2, n + 2):
        gdp[t] = (
            0.7 * gdp[t - 1]
            - 0.2 * gdp[t - 2]
            + 0.05 * np.random.randn()
        )
    gdp = gdp[2:]  # drop warm-up

    # -- Inflation: driven by GDP with a lag (Phillips curve) -----------------
    inflation = np.zeros(n)
    inflation[0] = 0.02 + 0.1 * np.random.randn()
    for t in range(1, n):
        drift = 0.01
        inflation[t] = (
            0.6 * inflation[t - 1]
            + 0.15 * gdp[t - 1]
            + drift
            + 0.08 * np.random.randn()
        )

    # -- Unemployment: inverse of GDP (Okun's law) ----------------------------
    unemployment = np.zeros(n)
    unemployment[0] = 5.0 + 0.2 * np.random.randn()
    for t in range(1, n):
        unemployment[t] = (
            0.7 * unemployment[t - 1]
            - 0.3 * gdp[t - 1]
            + 0.1  # natural rate drift
            + 0.15 * np.random.randn()
        )

    # Stack features: (n, 3)
    features = np.column_stack([gdp, inflation, unemployment])

    # Build sliding-window sequences
    n_samples = n - seq_len
    X = np.zeros((n_samples, seq_len, 3))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        X[i] = features[i : i + seq_len]
        y[i] = gdp[i + seq_len]  # next-period GDP

    return X, y


def generate_financial_data(
    n: int = 1500,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic financial return data with volatility clustering.

    Uses a GARCH(1,1)-like process to produce:

    - **returns** -- daily log returns with time-varying volatility
    - **volume** -- proxy for trading volume, correlated with |returns|
    - **realized_vol** -- 5-period rolling standard deviation of returns

    Volatility clustering means large returns tend to be followed by large
    returns (of either sign), mimicking real financial data.

    Parameters
    ----------
    n : int
        Number of time steps to generate.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n - seq_len, seq_len, 3)
        Input sequences of length 20 with 3 features.
    y : np.ndarray, shape (n - seq_len,)
        Target values (next-step returns).
    """
    if random_state is not None:
        np.random.seed(random_state)

    seq_len = 20

    # -- GARCH(1,1) volatility process ----------------------------------------
    omega = 0.01
    alpha = 0.1
    beta = 0.85

    returns = np.zeros(n)
    variance = np.ones(n) * omega / (1.0 - alpha - beta)

    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1]**2 + beta * variance[t - 1]
        returns[t] = np.sqrt(variance[t]) * np.random.randn()

    # -- Volume: correlated with absolute returns -----------------------------
    volume = np.zeros(n)
    base_vol = 1.0
    for t in range(n):
        volume[t] = max(0.1, base_vol + 2.0 * np.abs(returns[t]) + 0.3 * np.random.randn())

    # -- Realized volatility: rolling window ----------------------------------
    window = 5
    realized_vol = np.zeros(n)
    for t in range(window, n):
        realized_vol[t] = np.std(returns[t - window: t])

    # Stack features
    features = np.column_stack([returns, volume, realized_vol])

    # Build sliding windows
    n_samples = n - seq_len
    X = np.zeros((n_samples, seq_len, 3))
    y = np.zeros(n_samples)

    for i in range(n_samples):
        X[i] = features[i: i + seq_len]
        y[i] = returns[i + seq_len]

    return X, y



