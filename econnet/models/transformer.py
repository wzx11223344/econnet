"""
Time-series Transformer Forecaster -- pure NumPy implementation.

Implements a Transformer encoder for sequence-to-one time series
forecasting with sinusoidal positional encoding, multi-head self-attention,
feed-forward layers, and layer normalisation -- all manually computed.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Activation & normalisation helpers
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Layer Normalisation
# ---------------------------------------------------------------------------

class _LayerNorm:
    """Manual layer normalisation: y = gamma * (x - mu) / sigma + beta."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

        # Cache for backward pass
        self._cache: Dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        x : np.ndarray, shape (batch, seq_len, d_model)

        Returns
        -------
        np.ndarray, same shape as x
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std = np.sqrt(var + self.eps)

        x_norm = (x - mean) / std

        self._cache = {"x_norm": x_norm, "std": std}

        return self.gamma * x_norm + self.beta

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass.

        Returns
        -------
        d_x : np.ndarray
        d_gamma : np.ndarray
        d_beta : np.ndarray
        """
        x_norm = self._cache["x_norm"]
        std = self._cache["std"]
        D = d_out.shape[-1]

        d_beta = np.sum(d_out, axis=(0, 1))
        d_gamma = np.sum(d_out * x_norm, axis=(0, 1))

        dx_norm = d_out * self.gamma

        # Backward through (x - mean) / std
        d_std = np.sum(dx_norm * (x_norm / (-std)), axis=-1, keepdims=True)
        d_mean = np.sum(dx_norm * (-1.0 / std), axis=-1, keepdims=True)

        dx = dx_norm / std

        # Correction terms for mean and variance
        dx += (2.0 / D) * x_norm * d_std
        dx += d_mean / D

        return dx, d_gamma, d_beta


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

def _positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Generate sinusoidal positional encodings.

    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    Parameters
    ----------
    seq_len : int
        Sequence length.
    d_model : int
        Model dimension (must be even).

    Returns
    -------
    np.ndarray, shape (1, seq_len, d_model)
    """
    pos = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]  # (1, d_model)

    angle = pos / np.power(10000.0, (2.0 * (i // 2)) / d_model)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])

    return pe[np.newaxis, :, :]  # (1, seq_len, d_model)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class _MultiHeadSelfAttention:
    """Multi-head scaled dot-product self-attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        limit = np.sqrt(6.0 / (d_model + self.d_k))
        self.W_q = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_k = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_v = np.random.uniform(-limit, limit, (d_model, d_model))
        self.W_o = np.random.uniform(-limit, limit, (d_model, d_model))

        self.b_q = np.zeros(d_model)
        self.b_k = np.zeros(d_model)
        self.b_v = np.zeros(d_model)
        self.b_o = np.zeros(d_model)

        self._cache: Dict = {}

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (batch, seq, d_model) -> (batch, n_heads, seq, d_k)."""
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (batch, n_heads, seq, d_k) -> (batch, seq, d_model)."""
        batch, _, seq, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for multi-head self-attention."""
        batch, seq_len, _ = x.shape

        # Linear projections
        Q = x @ self.W_q + self.b_q  # (batch, seq, d_model)
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        # Split into heads
        Q_h = self._split_heads(Q)  # (batch, n_heads, seq, d_k)
        K_h = self._split_heads(K)
        V_h = self._split_heads(V)

        # Scaled dot-product attention
        scale = np.sqrt(self.d_k)
        scores = Q_h @ K_h.transpose(0, 1, 3, 2) / scale  # (batch, n_heads, seq, seq)
        attn_weights = _softmax(scores, axis=-1)
        attn_out = attn_weights @ V_h  # (batch, n_heads, seq, d_k)

        # Merge heads
        concat = self._merge_heads(attn_out)
        out = concat @ self.W_o + self.b_o  # (batch, seq, d_model)

        # Cache for backward
        self._cache = {
            "x": x,
            "Q": Q,
            "K": K,
            "V": V,
            "Q_h": Q_h,
            "K_h": K_h,
            "V_h": V_h,
            "scores": scores,
            "attn_weights": attn_weights,
            "attn_out": attn_out,
            "concat": concat,
        }

        return out

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Backward pass for multi-head self-attention.

        Returns
        -------
        d_x : np.ndarray
        param_grads : dict
        """
        c = self._cache
        batch, seq_len, d_model = d_out.shape

        # Gradient through output projection
        d_concat = d_out @ self.W_o.T  # (batch, seq, d_model)
        d_W_o = c["concat"].reshape(-1, d_model).T @ d_out.reshape(-1, d_model)
        d_b_o = np.sum(d_out.reshape(-1, d_model), axis=0)

        # Gradient through head merge
        d_attn_out = d_concat.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(
            0, 2, 1, 3
        )  # (batch, n_heads, seq, d_k)

        # Gradient through attention
        # d(attn_weights @ V_h) -> d_attn_weights, d_V_h
        d_attn_weights = d_attn_out @ c["V_h"].transpose(0, 1, 3, 2)
        d_V_h = c["attn_weights"].transpose(0, 1, 3, 2) @ d_attn_out

        # Gradient through softmax
        # d_scores = d_attn_weights * attn_weights - attn_weights * sum(d_attn_weights * attn_weights)
        attn = c["attn_weights"]
        d_scores = attn * (d_attn_weights - np.sum(d_attn_weights * attn, axis=-1, keepdims=True))
        d_scores = d_scores / np.sqrt(self.d_k)

        # Gradient through Q, K, V projections
        d_Q_h = d_scores @ c["K_h"]  # (batch, n_heads, seq, d_k)
        d_K_h = d_scores.transpose(0, 1, 3, 2) @ c["Q_h"]

        # Merge heads back
        d_Q = d_Q_h.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        d_K = d_K_h.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        d_V = d_V_h.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

        # Gradient through linear projections
        x_flat = c["x"].reshape(-1, d_model)

        d_W_q = x_flat.T @ d_Q.reshape(-1, d_model)
        d_W_k = x_flat.T @ d_K.reshape(-1, d_model)
        d_W_v = x_flat.T @ d_V.reshape(-1, d_model)
        d_b_q = np.sum(d_Q.reshape(-1, d_model), axis=0)
        d_b_k = np.sum(d_K.reshape(-1, d_model), axis=0)
        d_b_v = np.sum(d_V.reshape(-1, d_model), axis=0)

        # Gradient w.r.t. input x
        d_x = (
            d_Q @ self.W_q.T
            + d_K @ self.W_k.T
            + d_V @ self.W_v.T
        )

        param_grads = {
            "W_q": d_W_q,
            "W_k": d_W_k,
            "W_v": d_W_v,
            "W_o": d_W_o,
            "b_q": d_b_q,
            "b_k": d_b_k,
            "b_v": d_b_v,
            "b_o": d_b_o,
        }

        return d_x, param_grads


# ---------------------------------------------------------------------------
# Transformer Encoder Layer
# ---------------------------------------------------------------------------

class _TransformerEncoderLayer:
    """A single Transformer encoder layer: Attention + FFN, each with residual + LN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        self.attn = _MultiHeadSelfAttention(d_model, n_heads)
        self.ln1 = _LayerNorm(d_model)
        self.ln2 = _LayerNorm(d_model)

        # Feed-forward: d_model -> d_ff -> d_model
        limit = np.sqrt(6.0 / (d_model + d_ff))
        self.W1 = np.random.uniform(-limit, limit, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.uniform(-limit, limit, (d_ff, d_model))
        self.b2 = np.zeros(d_model)

        self._cache: Dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Self-attention sub-layer
        attn_out = self.attn.forward(x)
        x1 = self.ln1.forward(x + attn_out)

        # Feed-forward sub-layer
        ff_out = _relu(x1 @ self.W1 + self.b1) @ self.W2 + self.b2
        out = self.ln2.forward(x1 + ff_out)

        self._cache = {
            "x": x,
            "attn_out": attn_out,
            "x1": x1,
            "ff_hidden": _relu(x1 @ self.W1 + self.b1),
            "ff_out": ff_out,
        }

        return out

    def backward(self, d_out: np.ndarray) -> Tuple[np.ndarray, Dict, Dict]:
        """Backward pass.

        Returns
        -------
        d_x : np.ndarray
        attn_grads : dict
        ffn_grads : dict
        """
        c = self._cache

        # Backward through LN2
        d_x1_ff, d_gamma2, d_beta2 = self.ln2.backward(d_out)
        # Residual split
        d_ff_out = d_x1_ff
        d_x1_resid = d_x1_ff

        # Backward through FFN
        d_ff_hidden = d_ff_out @ self.W2.T
        d_W2 = c["ff_hidden"].reshape(-1, c["ff_hidden"].shape[-1]).T @ d_ff_out.reshape(-1, d_ff_out.shape[-1])
        d_b2 = np.sum(d_ff_out.reshape(-1, d_ff_out.shape[-1]), axis=0)

        d_pre_relu = d_ff_hidden * _relu_derivative(c["ff_hidden"])
        d_W1 = c["x1"].reshape(-1, c["x1"].shape[-1]).T @ d_pre_relu.reshape(-1, d_pre_relu.shape[-1])
        d_b1 = np.sum(d_pre_relu.reshape(-1, d_pre_relu.shape[-1]), axis=0)

        d_x1 = d_pre_relu @ self.W1.T + d_x1_resid

        # Backward through LN1
        d_x_attn, d_gamma1, d_beta1 = self.ln1.backward(d_x1)
        d_attn = d_x_attn
        d_x_resid = d_x_attn

        # Backward through attention
        d_x_attn_in, attn_grads = self.attn.backward(d_attn)
        d_x = d_x_attn_in + d_x_resid

        ffn_grads = {
            "W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2,
            "gamma1": d_gamma1, "beta1": d_beta1,
            "gamma2": d_gamma2, "beta2": d_beta2,
        }

        return d_x, attn_grads, ffn_grads


# ---------------------------------------------------------------------------
# Transformer Forecaster
# ---------------------------------------------------------------------------

class TransformerForecaster:
    """Time-series Transformer for sequence-to-one forecasting.

    Architecture
    ------------
    - Sinusoidal positional encoding added to input
    - Stack of Transformer encoder layers
    - Global average pooling over time
    - Linear projection to scalar output

    Parameters
    ----------
    d_model : int
        Model dimension (default 32).
    n_heads : int
        Number of attention heads (default 4).
    n_layers : int
        Number of encoder layers (default 2).
    d_ff : int
        Feed-forward hidden dimension (default 64).
    dropout : float
        Dropout probability (default 0.1).
    random_state : int, optional
        Seed for reproducibility.

    Examples
    --------
    >>> model = TransformerForecaster(d_model=32, n_heads=4, n_layers=2)
    >>> model.fit(X_train, y_train, epochs=50, lr=0.001)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 64,
        dropout: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.random_state = random_state

        self._layers: List[_TransformerEncoderLayer] = []
        self._W_in: Optional[np.ndarray] = None
        self._W_out: Optional[np.ndarray] = None
        self._b_out: Optional[np.ndarray] = None
        self._train_losses: List[float] = []

    def _init_params(self, input_size: int, seq_len: int) -> None:
        """Initialise all parameters."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Input projection: input_size -> d_model
        limit = np.sqrt(6.0 / (input_size + self.d_model))
        self._W_in = np.random.uniform(-limit, limit, (input_size, self.d_model))
        self._b_in = np.zeros(self.d_model)

        # Encoder layers
        self._layers = []
        for _ in range(self.n_layers):
            layer = _TransformerEncoderLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
            )
            self._layers.append(layer)

        # Output projection
        limit_out = np.sqrt(6.0 / (self.d_model + 1))
        self._W_out = np.random.uniform(-limit_out, limit_out, (self.d_model, 1))
        self._b_out = np.zeros((1, 1))

    def _forward(
        self, X: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """Forward pass.

        Parameters
        ----------
        X : np.ndarray, shape (batch, seq_len, features)
        training : bool

        Returns
        -------
        out : np.ndarray, shape (batch, 1)
        layer_caches : list of dict or None
        """
        _, seq_len, _ = X.shape

        # Input projection: (batch, seq, features) -> (batch, seq, d_model)
        h = X @ self._W_in + self._b_in

        # Add positional encoding
        pe = _positional_encoding(seq_len, self.d_model)
        h = h + pe

        layer_caches: List = [] if training else None

        # Encoder layers
        for layer in self._layers:
            h = layer.forward(h)
            if layer_caches is not None:
                layer_caches.append(layer._cache.copy())

        # Global average pooling over time
        pooled = np.mean(h, axis=1)  # (batch, d_model)

        # Output projection
        out = pooled @ self._W_out + self._b_out  # (batch, 1)

        if layer_caches is not None:
            layer_caches.append({"pooled": pooled, "h": h})

        return out, layer_caches

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
    ) -> "TransformerForecaster":
        """Train the transformer forecaster.

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

        if self._W_in is None:
            self._init_params(X.shape[2], X.shape[1])

        self._train_losses = []

        for epoch in range(epochs):
            y_pred, layer_caches = self._forward(X, training=True)

            loss = np.mean((y_pred - y) ** 2)
            self._train_losses.append(float(loss))

            # Backward
            batch_size = y_pred.shape[0]

            # Gradient of MSE
            d_out = (2.0 / batch_size) * (y_pred - y)  # (batch, 1)

            # Output projection gradients
            pooled = layer_caches[-1]["pooled"]
            d_W_out = pooled.T @ d_out
            d_b_out = np.sum(d_out, axis=0, keepdims=True)

            # Gradient through pooling
            d_pooled = d_out @ self._W_out.T  # (batch, d_model)
            h_final = layer_caches[-1]["h"]
            seq_len = h_final.shape[1]
            d_h = np.zeros_like(h_final)
            d_h[:] = d_pooled[:, np.newaxis, :] / seq_len

            # Backward through layers
            for layer in reversed(self._layers):
                d_h, attn_grads, ffn_grads = layer.backward(d_h)

                # Update layer parameters
                layer.attn.W_q -= lr * attn_grads["W_q"]
                layer.attn.W_k -= lr * attn_grads["W_k"]
                layer.attn.W_v -= lr * attn_grads["W_v"]
                layer.attn.W_o -= lr * attn_grads["W_o"]
                layer.attn.b_q -= lr * attn_grads["b_q"]
                layer.attn.b_k -= lr * attn_grads["b_k"]
                layer.attn.b_v -= lr * attn_grads["b_v"]
                layer.attn.b_o -= lr * attn_grads["b_o"]

                layer.W1 -= lr * ffn_grads["W1"]
                layer.b1 -= lr * ffn_grads["b1"]
                layer.W2 -= lr * ffn_grads["W2"]
                layer.b2 -= lr * ffn_grads["b2"]
                layer.ln1.gamma -= lr * ffn_grads["gamma1"]
                layer.ln1.beta -= lr * ffn_grads["beta1"]
                layer.ln2.gamma -= lr * ffn_grads["gamma2"]
                layer.ln2.beta -= lr * ffn_grads["beta2"]

            # Update output projection
            self._W_out -= lr * d_W_out
            self._b_out -= lr * d_b_out

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
        if self._W_in is not None:
            total += int(self._W_in.size + self._b_in.size)
        for layer in self._layers:
            total += int(
                layer.attn.W_q.size + layer.attn.W_k.size + layer.attn.W_v.size
                + layer.attn.W_o.size
                + layer.attn.b_q.size + layer.attn.b_k.size
                + layer.attn.b_v.size + layer.attn.b_o.size
                + layer.W1.size + layer.b1.size
                + layer.W2.size + layer.b2.size
                + layer.ln1.gamma.size + layer.ln1.beta.size
                + layer.ln2.gamma.size + layer.ln2.beta.size
            )
        if self._W_out is not None:
            total += int(self._W_out.size + self._b_out.size)
        return total
