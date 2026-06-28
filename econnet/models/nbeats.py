"""
N-BEATS Forecaster -- pure NumPy implementation.

Implements the Neural Basis Expansion Analysis for interpretable time
series forecasting.  Uses trend (polynomial) and seasonality (Fourier)
basis functions with doubly residual stacking topology.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------

def _trend_basis(t: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial trend basis: [t^0, t^1, ..., t^{degree}] (normalised).

    Parameters
    ----------
    t : np.ndarray, shape (horizon,) or (backcast_length,)
        Time indices (0, 1, 2, ...).
    degree : int
        Polynomial degree.

    Returns
    -------
    np.ndarray, shape (len(t), degree + 1)
    """
    t = t.astype(np.float64)
    t_norm = t / (np.max(t) + 1e-8)
    basis = np.column_stack([t_norm ** p for p in range(degree + 1)])
    # Normalise columns for numerical stability
    norms = np.linalg.norm(basis, axis=0, keepdims=True) + 1e-8
    return basis / norms


def _seasonality_basis(t: np.ndarray, n_harmonics: int, period: float) -> np.ndarray:
    """Fourier seasonality basis.

    Parameters
    ----------
    t : np.ndarray, shape (horizon,) or (backcast_length,)
        Time indices.
    n_harmonics : int
        Number of Fourier harmonics.
    period : float
        Base period (e.g., 7 for weekly, 365.25 for yearly).

    Returns
    -------
    np.ndarray, shape (len(t), 2 * n_harmonics)
    """
    t = t.astype(np.float64)
    t_norm = 2.0 * np.pi * t / period
    basis_parts = []
    for h in range(1, n_harmonics + 1):
        basis_parts.append(np.sin(h * t_norm))
        basis_parts.append(np.cos(h * t_norm))
    basis = np.column_stack(basis_parts)
    norms = np.linalg.norm(basis, axis=0, keepdims=True) + 1e-8
    return basis / norms


# ---------------------------------------------------------------------------
# N-BEATS Block
# ---------------------------------------------------------------------------

class _NBEATSBlock:
    """A single N-BEATS block with basis expansion.

    The block maps the input (lookback window) through several FC layers,
    then projects to basis coefficients which are used to construct
    both a backcast (of the input) and a forecast (of the future).
    """

    def __init__(
        self,
        input_size: int,
        theta_dim: int,
        n_layers: int,
        hidden_size: int,
        backcast_length: int,
        forecast_length: int,
        trend_degree: int,
        n_harmonics: int,
        period: float,
    ) -> None:
        self.input_size = input_size
        self.theta_dim = theta_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        # Trend basis dimension
        self.trend_dim = trend_degree + 1
        # Seasonality basis dimension
        self.seasonal_dim = 2 * n_harmonics
        self.basis_dim = self.trend_dim + self.seasonal_dim

        # Build FC layers
        self._fc_weights: List[np.ndarray] = []
        self._fc_biases: List[np.ndarray] = []

        dims = [input_size] + [hidden_size] * n_layers + [theta_dim]

        for i in range(len(dims) - 1):
            limit = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
            self._fc_weights.append(
                np.random.uniform(-limit, limit, (dims[i], dims[i + 1]))
            )
            self._fc_biases.append(np.zeros(dims[i + 1]))

        # Backcast basis
        t_backcast = np.arange(backcast_length)
        trend_b = _trend_basis(t_backcast, trend_degree)
        season_b = _seasonality_basis(t_backcast, n_harmonics, period)
        self._backcast_basis = np.hstack([trend_b, season_b])  # (backcast_len, basis_dim)

        # Forecast basis
        t_forecast = np.arange(backcast_length, backcast_length + forecast_length)
        trend_f = _trend_basis(t_forecast, trend_degree)
        season_f = _seasonality_basis(t_forecast, n_harmonics, period)
        self._forecast_basis = np.hstack([trend_f, season_f])  # (forecast_len, basis_dim)

        # Cache
        self._cache: Dict = {}

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass.

        Parameters
        ----------
        x : np.ndarray, shape (batch, backcast_length)

        Returns
        -------
        backcast : np.ndarray, shape (batch, backcast_length)
        forecast : np.ndarray, shape (batch, forecast_length)
        """
        batch = x.shape[0]

        # FC stack
        h = x
        fc_activations = []
        for w, b in zip(self._fc_weights, self._fc_biases):
            h_pre = h @ w + b
            fc_activations.append(h_pre)
            h = np.maximum(0.0, h_pre)  # ReLU

        # Theta: coefficients for basis expansion
        theta = h  # (batch, theta_dim)

        # Split theta into backcast and forecast coefficients
        # We use theta_dim = 2 * basis_dim in the canonical N-BEATS
        # But for simplicity we split: first half for backcast, second for forecast
        theta_b = theta[:, : self.theta_dim // 2]
        theta_f = theta[:, self.theta_dim // 2:]

        # Ensure theta_b and theta_f have right dimensions
        if theta_b.shape[1] != self.basis_dim:
            # Pad or truncate
            if theta_b.shape[1] < self.basis_dim:
                pad = np.zeros((batch, self.basis_dim - theta_b.shape[1]))
                theta_b = np.hstack([theta_b, pad])
                theta_f = np.hstack([theta_f, np.zeros((batch, self.basis_dim - theta_f.shape[1]))])
            else:
                theta_b = theta_b[:, : self.basis_dim]
                theta_f = theta_f[:, : self.basis_dim]

        # Basis projection
        backcast = theta_b @ self._backcast_basis.T  # (batch, backcast_len)
        forecast = theta_f @ self._forecast_basis.T  # (batch, forecast_len)

        self._cache = {
            "x": x,
            "fc_activations": fc_activations,
            "theta": theta,
            "theta_b": theta_b,
            "theta_f": theta_f,
        }

        return backcast, forecast

    def backward(
        self, d_backcast: np.ndarray, d_forecast: np.ndarray
    ) -> np.ndarray:
        """Backward pass.

        Parameters
        ----------
        d_backcast : np.ndarray, shape (batch, backcast_length)
        d_forecast : np.ndarray, shape (batch, forecast_length)

        Returns
        -------
        d_x : np.ndarray, shape (batch, input_size)
        """
        c = self._cache
        batch = d_backcast.shape[0]

        # Gradient through basis projections
        d_theta_b = d_backcast @ self._backcast_basis  # (batch, basis_dim)
        d_theta_f = d_forecast @ self._forecast_basis  # (batch, basis_dim)

        # Combine theta gradients
        half_dim = self.theta_dim // 2
        d_theta = np.zeros((batch, self.theta_dim))

        actual_b_dim = min(half_dim, self.basis_dim)
        d_theta[:, :actual_b_dim] = d_theta_b[:, :actual_b_dim]
        d_theta[:, half_dim:half_dim + actual_b_dim] = d_theta_f[:, :actual_b_dim]

        # Gradient through FC layers (ReLU)
        d_h = d_theta
        grads_w: List[np.ndarray] = []
        grads_b: List[np.ndarray] = []

        n_fc = len(self._fc_weights)
        for i in range(n_fc - 1, -1, -1):
            h_pre = c["fc_activations"][i]

            # ReLU backward
            d_h = d_h * (h_pre > 0).astype(np.float64)

            # Weight/bias gradients
            if i > 0:
                x_in = np.maximum(0.0, c["fc_activations"][i - 1])
            else:
                x_in = c["x"]

            grads_w.insert(0, x_in.T @ d_h)
            grads_b.insert(0, np.sum(d_h, axis=0))

            d_h = d_h @ self._fc_weights[i].T

        # Store gradients for update
        self._d_weights = grads_w
        self._d_biases = grads_b

        return d_h

    def apply_gradients(self, lr: float) -> None:
        """Apply accumulated gradients with learning rate."""
        for i in range(len(self._fc_weights)):
            self._fc_weights[i] -= lr * self._d_weights[i]
            self._fc_biases[i] -= lr * self._d_biases[i]


# ---------------------------------------------------------------------------
# N-BEATS Forecaster
# ---------------------------------------------------------------------------

class NBEATSForecaster:
    """N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting.

    Architecture
    ------------
    Stack of interpretable blocks, each decomposing the signal into
    trend (polynomial) and seasonality (Fourier) components via basis
    expansion.  Uses doubly residual connections.

    Parameters
    ----------
    n_blocks : int
        Number of N-BEATS blocks (default 3).
    n_layers : int
        Number of FC layers per block (default 4).
    hidden_size : int
        Hidden dimension of FC layers (default 64).
    theta_dim : int
        Dimension of the expansion coefficient vector (default 64).
    trend_degree : int
        Polynomial degree for trend basis (default 3).
    n_harmonics : int
        Number of Fourier harmonics for seasonality (default 4).
    period : float
        Base period for seasonality (default 7.0).
    random_state : int, optional
        Seed for reproducibility.

    Examples
    --------
    >>> model = NBEATSForecaster(n_blocks=3, hidden_size=64)
    >>> model.fit(X_train, y_train, epochs=50, lr=0.001)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        n_blocks: int = 3,
        n_layers: int = 4,
        hidden_size: int = 64,
        theta_dim: int = 64,
        trend_degree: int = 3,
        n_harmonics: int = 4,
        period: float = 7.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.theta_dim = theta_dim
        self.trend_degree = trend_degree
        self.n_harmonics = n_harmonics
        self.period = period
        self.random_state = random_state

        self._blocks: List[_NBEATSBlock] = []
        self._train_losses: List[float] = []

    def _init_params(self, backcast_length: int, forecast_length: int) -> None:
        """Initialise N-BEATS blocks."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._blocks = []
        for _ in range(self.n_blocks):
            block = _NBEATSBlock(
                input_size=backcast_length,
                theta_dim=self.theta_dim,
                n_layers=self.n_layers,
                hidden_size=self.hidden_size,
                backcast_length=backcast_length,
                forecast_length=forecast_length,
                trend_degree=self.trend_degree,
                n_harmonics=self.n_harmonics,
                period=self.period,
            )
            self._blocks.append(block)

    def _forward(
        self, X: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, Optional[List]]:
        """Forward pass through stacked blocks.

        Parameters
        ----------
        X : np.ndarray, shape (batch, seq_len, n_features)
            Input sequences.  For N-BEATS we flatten the last two dims.

        Returns
        -------
        forecast : np.ndarray, shape (batch, forecast_length)
        block_caches : list or None
        """
        batch, seq_len, n_features = X.shape

        # Flatten: (batch, seq_len * n_features)
        x = X.reshape(batch, -1)

        forecast_total = np.zeros((batch, self._blocks[0].forecast_length))
        block_caches = [] if training else None

        for block in self._blocks:
            backcast, forecast = block.forward(x)

            # Doubly residual: x = x - backcast, forecast_total += forecast
            x = x - backcast
            forecast_total = forecast_total + forecast

            if block_caches is not None:
                block_caches.append(block._cache.copy())

        return forecast_total, block_caches

    def _backward(
        self,
        block_caches: List[Dict],
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """Backward pass through all blocks.

        Stores gradients internally in each block.
        """
        batch = y_pred.shape[0]
        seq_len, n_features = X.shape[1], X.shape[2]
        backcast_length = seq_len * n_features
        forecast_length = y_pred.shape[1]

        # Gradient of MSE w.r.t. forecast
        d_forecast = (2.0 / batch) * (y_pred - y)  # (batch, forecast_length)

        d_x = np.zeros((batch, backcast_length))

        # Backward through blocks in reverse
        for i in range(self.n_blocks - 1, -1, -1):
            block = self._blocks[i]

            # For the top block, d_forecast comes directly from loss
            # For lower blocks, d_forecast is the same (sum of forecasts)
            # d_backcast = -d_x (from residual: x_new = x_old - backcast)
            d_backcast = -d_x if i < self.n_blocks - 1 else np.zeros((batch, backcast_length))

            # For the last block, d_x comes from the residual update
            # Actually the structure is: x_0 = X; for each block: backcast_i, forecast_i = block_i(x_{i-1}); x_i = x_{i-1} - backcast_i; total_forecast += forecast_i
            # d_loss/d_x_{i-1} = d_loss/d_x_i + d_loss/d_backcast_i * (-1)
            # d_loss/d_backcast_i = d_loss/d_x_i * (-1) since x_i = x_{i-1} - backcast_i

            if i == self.n_blocks - 1:
                d_backcast = -d_x  # 0 initially, so d_backcast = 0

            d_x_in = block.backward(d_backcast, d_forecast)
            d_x = d_x_in + d_x  # accumulate gradient for previous block

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.001,
        verbose: bool = True,
    ) -> "NBEATSForecaster":
        """Train the N-BEATS forecaster.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, seq_len, n_features)
        y : np.ndarray, shape (n_samples,)
        epochs : int
        lr : float
        verbose : bool

        Returns
        -------
        self
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        forecast_length = y.shape[1]
        backcast_length = X.shape[1] * X.shape[2]

        if not self._blocks:
            self._init_params(backcast_length, forecast_length)

        self._train_losses = []

        for epoch in range(epochs):
            # Forward
            y_pred, block_caches = self._forward(X, training=True)
            loss = np.mean((y_pred - y) ** 2)
            self._train_losses.append(float(loss))

            # Backward
            self._backward(block_caches, X, y, y_pred)

            # Apply gradients
            for block in self._blocks:
                block.apply_gradients(lr)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1:4d}/{epochs}  loss = {loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate forecasts.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, seq_len, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        y_pred, _ = self._forward(X, training=False)
        return y_pred.ravel()

    def count_params(self) -> int:
        """Return the total number of trainable parameters."""
        total = 0
        for block in self._blocks:
            for w in block._fc_weights:
                total += int(w.size)
            for b in block._fc_biases:
                total += int(b.size)
        return total
