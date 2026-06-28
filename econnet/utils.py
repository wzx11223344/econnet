"""
Utility functions for EconNet.

Provides train/test splitting, evaluation metrics, and data normalization
utilities used across all models.
"""

import numpy as np
from typing import Tuple, Optional, Union


# ---------------------------------------------------------------------------
# Train / Test Split
# ---------------------------------------------------------------------------

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = False,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split time-series data into training and test sets.

    For time-series data, shuffling is disabled by default to preserve
    temporal ordering.  The split is always chronological (earliest data
    goes to training).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, seq_len, n_features)
        Input sequences.
    y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)
        Target values.
    test_size : float
        Fraction of data to use for testing (default 0.2).
    shuffle : bool
        Whether to shuffle before splitting (default False).
    random_state : int, optional
        Seed for the random number generator.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of np.ndarray
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(n * (1.0 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MSE = mean((y_true - y_pred)^2)
    """
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAE = mean(|y_true - y_pred|)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSE = sqrt(mean((y_true - y_pred)^2))
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        MAPE = 100 * mean(|(y_true - y_pred) / (y_true + eps)|)
    """
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (y_true + eps))))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class StandardScaler:
    """Standard (z-score) normalizer.

    Transforms data to zero mean and unit variance:  x' = (x - mu) / sigma

    Examples
    --------
    >>> scaler = StandardScaler()
    >>> X_train = scaler.fit_transform(X_train)
    >>> X_test = scaler.transform(X_test)
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute mean and standard deviation from X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        self.mean_ = np.mean(X, axis=0, keepdims=True)
        self.std_ = np.std(X, axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization to X.

        Parameters
        ----------
        X : np.ndarray
            Data to normalize.

        Returns
        -------
        np.ndarray
            Standardized data.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler has not been fitted. Call fit() first.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data and return the transformed version.

        Parameters
        ----------
        X : np.ndarray
            Data to fit and transform.

        Returns
        -------
        np.ndarray
            Standardized data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the standardization:  x = x' * sigma + mu

        Parameters
        ----------
        X : np.ndarray
            Standardized data.

        Returns
        -------
        np.ndarray
            Original-scale data.
        """
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler has not been fitted. Call fit() first.")
        return X * self.std_ + self.mean_


class MinMaxScaler:
    """Min-Max normalizer.

    Transforms data to [0, 1]:  x' = (x - min) / (max - min)

    Examples
    --------
    >>> scaler = MinMaxScaler()
    >>> X_train = scaler.fit_transform(X_train)
    >>> X_test = scaler.transform(X_test)
    """

    def __init__(self) -> None:
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Compute min and max from X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self
        """
        self.min_ = np.min(X, axis=0, keepdims=True)
        self.max_ = np.max(X, axis=0, keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply min-max scaling to X.

        Parameters
        ----------
        X : np.ndarray
            Data to normalize.

        Returns
        -------
        np.ndarray
            Scaled data in [0, 1].
        """
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("MinMaxScaler has not been fitted. Call fit() first.")
        denom = self.max_ - self.min_ + 1e-8
        return (X - self.min_) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data and return the transformed version."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse min-max scaling."""
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("MinMaxScaler has not been fitted. Call fit() first.")
        return X * (self.max_ - self.min_ + 1e-8) + self.min_


# ---------------------------------------------------------------------------
# Sliding-window helper
# ---------------------------------------------------------------------------

def create_sequences(
    data: np.ndarray, seq_len: int, horizon: int = 1, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Create input-target pairs from a time series using a sliding window.

    Parameters
    ----------
    data : np.ndarray, shape (n_timesteps, n_features) or (n_timesteps,)
        1-D or 2-D time series.
    seq_len : int
        Number of past time steps used as input.
    horizon : int
        Number of future time steps to predict.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    X : np.ndarray, shape (n_samples, seq_len, n_features)
    y : np.ndarray, shape (n_samples, horizon)
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n = data.shape[0]
    n_features = data.shape[1]
    n_samples = (n - seq_len - horizon + 1) // stride

    X = np.zeros((n_samples, seq_len, n_features))
    y = np.zeros((n_samples, horizon))

    for i in range(n_samples):
        start = i * stride
        end_input = start + seq_len
        end_target = end_input + horizon
        X[i] = data[start:end_input, :]
        y[i] = data[end_input:end_target, 0]  # predict first feature

    return X, y


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model) -> int:
    """Count the total number of trainable scalar parameters in a model.

    Walks the model's ``__dict__`` and sums the sizes of all numpy arrays.

    Parameters
    ----------
    model : object
        A model instance whose attributes include numpy ndarrays.

    Returns
    -------
    int
        Total parameter count.
    """
    total = 0
    for name, value in model.__dict__.items():
        if isinstance(value, np.ndarray):
            total += int(value.size)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, np.ndarray):
                    total += int(item.size)
    return total
