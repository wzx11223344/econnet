"""
Temporal Convolutional Network (TCN) Forecaster -- pure NumPy implementation.

Implements a TCN with dilated causal convolutions and residual blocks for
sequence-to-one time series forecasting.  All convolution operations are
performed via manual nested loops -- zero deep learning framework dependency.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit."""
    return np.maximum(0.0, x)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if x > 0 else 0."""
    return (x > 0).astype(np.float64)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _he_init(shape: Tuple[int, ...]) -> np.ndarray:
    """He (Kaiming) normal initialisation for ReLU activations."""
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


# ---------------------------------------------------------------------------
# Manual 1-D Dilated Causal Convolution
# ---------------------------------------------------------------------------

def _dilated_causal_conv1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    dilation: int,
    training: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """Manual 1-D dilated causal convolution.

    For an input sequence x of shape (batch, channels, seq_len), the
    convolution at position t (0-indexed) is:

        out[t] = sum_{k=0}^{K-1} W[:, :, k] @ x[:, t - dilation * k] + b

    The causal constraint ensures only past (t - d*k >= 0) positions are used.

    Parameters
    ----------
    x : np.ndarray, shape (batch, in_channels, seq_len)
    weight : np.ndarray, shape (out_channels, in_channels, kernel_size)
    bias : np.ndarray, shape (out_channels,)
    dilation : int
        Dilation factor.
    training : bool
        If True, cache intermediate values for backward pass.

    Returns
    -------
    out : np.ndarray, shape (batch, out_channels, seq_len)
    cache : dict (for backward pass)
    """
    batch, in_c, seq_len = x.shape
    out_c, _, kernel_size = weight.shape

    out = np.zeros((batch, out_c, seq_len))
    cache: Dict = {
        "x": x,
        "weight": weight,
        "dilation": dilation,
    } if training else {}

    for b in range(batch):
        for oc in range(out_c):
            for t in range(seq_len):
                val = bias[oc]
                for ic in range(in_c):
                    for k in range(kernel_size):
                        src_idx = t - dilation * k
                        if src_idx >= 0:
                            val += weight[oc, ic, k] * x[b, ic, src_idx]
                out[b, oc, t] = val

    return out, cache


def _dilated_causal_conv1d_backward(
    d_out: np.ndarray, cache: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass for dilated causal convolution.

    Returns
    -------
    d_x : np.ndarray, shape (batch, in_channels, seq_len)
    d_w : np.ndarray, shape (out_channels, in_channels, kernel_size)
    d_b : np.ndarray, shape (out_channels,)
    """
    x = cache["x"]
    weight = cache["weight"]
    dilation = cache["dilation"]

    batch, in_c, seq_len = x.shape
    out_c, _, kernel_size = weight.shape

    d_x = np.zeros_like(x)
    d_w = np.zeros_like(weight)
    d_b = np.zeros(out_c)

    for b in range(batch):
        for oc in range(out_c):
            for t in range(seq_len):
                grad = d_out[b, oc, t]
                d_b[oc] += grad
                for ic in range(in_c):
                    for k in range(kernel_size):
                        src_idx = t - dilation * k
                        if src_idx >= 0:
                            d_w[oc, ic, k] += grad * x[b, ic, src_idx]
                            d_x[b, ic, src_idx] += grad * weight[oc, ic, k]

    return d_x, d_w, d_b


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class _ResidualBlock:
    """A single TCN residual block.

    Contains two dilated causal conv layers with ReLU activations and
    a 1x1 conv for residual projection when channel dimensions change.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout

        # Conv1: in_c -> out_c
        self.W1 = _he_init((out_channels, in_channels, kernel_size))
        self.b1 = np.zeros(out_channels)

        # Conv2: out_c -> out_c
        self.W2 = _he_init((out_channels, out_channels, kernel_size))
        self.b2 = np.zeros(out_channels)

        # 1x1 projection if channels differ
        if in_channels != out_channels:
            self.W_proj = _he_init((out_channels, in_channels, 1))
            self.b_proj = np.zeros(out_channels)
        else:
            self.W_proj = None
            self.b_proj = None

    def forward(
        self, x: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """Forward pass through the residual block.

        Parameters
        ----------
        x : np.ndarray, shape (batch, in_channels, seq_len)
        training : bool

        Returns
        -------
        out : np.ndarray, shape (batch, out_channels, seq_len)
        cache : dict
        """
        # ---- Conv1 → ReLU → Dropout ----
        h1, cache1 = _dilated_causal_conv1d(
            x, self.W1, self.b1, self.dilation, training
        )
        h1_act = _relu(h1)

        if training and self.dropout > 0:
            mask1 = (np.random.rand(*h1_act.shape) > self.dropout).astype(np.float64)
            mask1 /= (1.0 - self.dropout)
            h1_act = h1_act * mask1
        else:
            mask1 = None

        # ---- Conv2 → ReLU → Dropout ----
        h2, cache2 = _dilated_causal_conv1d(
            h1_act, self.W2, self.b2, self.dilation, training
        )
        h2_act = _relu(h2)

        if training and self.dropout > 0:
            mask2 = (np.random.rand(*h2_act.shape) > self.dropout).astype(np.float64)
            mask2 /= (1.0 - self.dropout)
            h2_act = h2_act * mask2
        else:
            mask2 = None

        # ---- Residual connection ----
        if self.W_proj is not None:
            # 1x1 conv for residual
            resid = np.zeros((x.shape[0], self.out_channels, x.shape[2]))
            for b in range(x.shape[0]):
                for oc in range(self.out_channels):
                    for t in range(x.shape[2]):
                        val = self.b_proj[oc]
                        for ic in range(self.in_channels):
                            val += self.W_proj[oc, ic, 0] * x[b, ic, t]
                        resid[b, oc, t] = val
        else:
            resid = x

        out = h2_act + resid

        cache: Dict = {
            "x": x,
            "h1": h1,
            "h1_act": h1_act,
            "h2": h2,
            "h2_act": h2_act,
            "resid": resid,
            "cache1": cache1,
            "cache2": cache2,
            "mask1": mask1,
            "mask2": mask2,
        } if training else {}

        return out, cache

    def backward(self, d_out: np.ndarray, cache: Dict) -> Tuple[np.ndarray, Dict]:
        """Backward pass through the residual block.

        Returns
        -------
        d_x : np.ndarray
            Gradient w.r.t. input.
        param_grads : dict
            Gradients for W1, b1, W2, b2, W_proj, b_proj.
        """
        # Residual gradient splits between h2_act and resid
        d_h2_act = d_out.copy()
        d_resid = d_out.copy()

        # ---- Backward through Conv2 ----
        if cache["mask2"] is not None and cache.get("training", True):
            d_h2_act = d_h2_act * cache["mask2"]

        d_h2 = d_h2_act * _relu_derivative(cache["h2"])
        d_h1_act, d_W2, d_b2 = _dilated_causal_conv1d_backward(d_h2, cache["cache2"])

        # ---- Backward through Conv1 ----
        if cache["mask1"] is not None and cache.get("training", True):
            d_h1_act = d_h1_act * cache["mask1"]

        d_h1 = d_h1_act * _relu_derivative(cache["h1"])
        d_x_from_conv, d_W1, d_b1 = _dilated_causal_conv1d_backward(
            d_h1, cache["cache1"]
        )

        # ---- Residual projection gradient ----
        grads: Dict = {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}

        if self.W_proj is not None:
            d_W_proj = np.zeros_like(self.W_proj)
            d_b_proj = np.zeros_like(self.b_proj)
            d_x_from_resid = np.zeros_like(cache["x"])
            x_src = cache["x"]

            for b in range(x_src.shape[0]):
                for oc in range(self.out_channels):
                    for t in range(x_src.shape[2]):
                        grad = d_resid[b, oc, t]
                        d_b_proj[oc] += grad
                        for ic in range(self.in_channels):
                            d_W_proj[oc, ic, 0] += grad * x_src[b, ic, t]
                            d_x_from_resid[b, ic, t] += (
                                grad * self.W_proj[oc, ic, 0]
                            )
            grads["W_proj"] = d_W_proj
            grads["b_proj"] = d_b_proj
            d_x = d_x_from_conv + d_x_from_resid
        else:
            d_x = d_x_from_conv + d_resid

        return d_x, grads


# ---------------------------------------------------------------------------
# TCN Forecaster
# ---------------------------------------------------------------------------

class TCNForecaster:
    """Temporal Convolutional Network for sequence-to-one forecasting.

    Architecture
    ------------
    A stack of residual blocks with exponentially increasing dilation
    (1, 2, 4, 8, ...).  After the final block the last time-step output
    is projected through a linear layer to produce the scalar forecast.

    Parameters
    ----------
    num_channels : list of int
        Number of output channels for each layer (default [16, 16, 16]).
    kernel_size : int
        Convolution kernel size (default 3).
    dropout : float
        Dropout probability (default 0.1).
    random_state : int, optional
        Seed for reproducibility.

    Examples
    --------
    >>> model = TCNForecaster(num_channels=[16, 16, 16], kernel_size=3, dropout=0.1)
    >>> model.fit(X_train, y_train, epochs=50, lr=0.001)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        if num_channels is None:
            num_channels = [16, 16, 16]
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.random_state = random_state

        # Built during fit
        self._blocks: List[_ResidualBlock] = []
        self._W_out: Optional[np.ndarray] = None
        self._b_out: Optional[np.ndarray] = None
        self._input_size: Optional[int] = None
        self._train_losses: List[float] = []

    def _init_params(self, input_size: int) -> None:
        """Initialise residual blocks and output projection."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self._input_size = input_size
        self._blocks = []

        in_c = input_size
        for layer_idx, out_c in enumerate(self.num_channels):
            dilation = 2 ** layer_idx
            block = _ResidualBlock(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout=self.dropout,
            )
            self._blocks.append(block)
            in_c = out_c

        # Output: (batch, out_c)  ->  (batch, 1)
        final_channels = self.num_channels[-1]
        self._W_out = _he_init((final_channels, 1)) * 0.1
        self._b_out = np.zeros((1,))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward(
        self, X: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """Forward pass through all residual blocks.

        Parameters
        ----------
        X : np.ndarray, shape (batch, seq_len, features)
        training : bool

        Returns
        -------
        out : np.ndarray, shape (batch, 1)
        caches : list of dict or None
        """
        # (batch, seq_len, features) -> (batch, features, seq_len)
        x = X.transpose(0, 2, 1)
        batch_size = x.shape[0]

        caches: List[Dict] = [] if training else None

        for block in self._blocks:
            h, block_cache = block.forward(x, training=training)
            x = h
            if caches is not None:
                caches.append(block_cache)

        # Global average pooling over time dimension
        pooled = np.mean(x, axis=2)  # (batch, out_c)

        # Output projection
        out = pooled @ self._W_out + self._b_out  # (batch, 1)

        return out, caches

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def _backward(
        self, caches: List[Dict], y: np.ndarray, y_pred: np.ndarray
    ) -> List[Dict]:
        """Backward pass through all residual blocks.

        Returns
        -------
        block_grads : list of dict
            Gradients for each block's parameters.
        """
        batch_size = y_pred.shape[0]
        seq_len = caches[0]["x"].shape[2]

        # Gradient of MSE loss
        d_out = (2.0 / batch_size) * (y_pred - y)  # (batch, 1)

        # Gradient through output projection
        d_pooled = d_out @ self._W_out.T  # (batch, out_c)
        d_W_out = d_pooled.T @ np.ones((batch_size, 1))  # simplified; use actual pooled
        # Actually, pooled = mean(x, axis=2), so the gradient is d_pooled / seq_len broadcast

        # Better: compute gradient of output projection correctly
        final_channels = self.num_channels[-1]
        x_final = caches[-1]["x"] if len(caches) > 0 else None
        if x_final is None:
            # Should not happen
            return []

        # We're computing gradient on the output of the last block
        # The output projection is applied after global avg pooling
        # So d_loss/d_W_out = pooled^T @ d_out, etc.
        # We'll handle this in the block backward loop

        block_grads: List[Dict] = []

        # Gradient flowing into the last block after pooling + linear
        # d_loss/d(last_block_output) = d_pooled / seq_len (broadcast)
        d_x = np.zeros((batch_size, final_channels, seq_len))
        d_pooled_unpooled = d_out @ self._W_out.T  # (batch, final_channels)
        d_x[:] = d_pooled_unpooled[:, :, np.newaxis] / seq_len

        # Backward through blocks in reverse
        for block, cache in zip(reversed(self._blocks), reversed(caches)):
            d_x, grads = block.backward(d_x, cache)
            block_grads.insert(0, grads)

        # Store output projection gradients for update
        self._d_W_out = np.zeros_like(self._W_out)
        self._d_b_out = np.zeros_like(self._b_out)

        # pooled is the actual average of the last block output
        final_output = caches[-1].get("h2_act", caches[-1].get("x"))
        # Actually let's recompute pooled from the last cache's x after the block forward
        # Since the architecture may have changed, compute using the actual output

        last_x = None
        for cache in reversed(caches):
            if "h2_act" in cache:
                last_x = cache["h2_act"]
                break
        if last_x is None:
            last_x = caches[-1]["x"]

        pooled = np.mean(last_x, axis=2)  # (batch, final_channels)
        self._d_W_out = pooled.T @ d_out
        self._d_b_out = np.sum(d_out, axis=0)

        return block_grads

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
    ) -> "TCNForecaster":
        """Train the TCN forecaster.

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

        if self._input_size is None:
            self._init_params(X.shape[2])

        self._train_losses = []

        for epoch in range(epochs):
            y_pred, caches = self._forward(X, training=True)
            loss = np.mean((y_pred - y) ** 2)
            self._train_losses.append(float(loss))

            block_grads = self._backward(caches, y, y_pred)

            # Update output projection
            self._W_out -= lr * self._d_W_out
            self._b_out -= lr * self._d_b_out

            # Update block parameters
            for block, grads in zip(self._blocks, block_grads):
                block.W1 -= lr * grads["W1"]
                block.b1 -= lr * grads["b1"]
                block.W2 -= lr * grads["W2"]
                block.b2 -= lr * grads["b2"]
                if "W_proj" in grads:
                    block.W_proj -= lr * grads["W_proj"]
                    block.b_proj -= lr * grads["b_proj"]

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
            total += int(block.W1.size + block.b1.size + block.W2.size + block.b2.size)
            if block.W_proj is not None:
                total += int(block.W_proj.size + block.b_proj.size)
        if self._W_out is not None:
            total += int(self._W_out.size + self._b_out.size)
        return total
