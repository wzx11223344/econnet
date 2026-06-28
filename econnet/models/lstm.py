"""
Stacked LSTM Forecaster -- pure NumPy implementation with manual BPTT.

Implements a multi-layer Long Short-Term Memory network for sequence-to-one
time series forecasting, with all forward/backward computations done manually
using NumPy.  No PyTorch or TensorFlow dependency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    # Clip to avoid overflow
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_derivative(s: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given the sigmoid output s."""
    return s * (1.0 - s)


def _tanh_derivative(t: np.ndarray) -> np.ndarray:
    """Derivative of tanh given the tanh output t."""
    return 1.0 - t * t


# ---------------------------------------------------------------------------
# Xavier / He initialisation
# ---------------------------------------------------------------------------

def _xavier_init(shape: Tuple[int, ...]) -> np.ndarray:
    """Xavier (Glorot) uniform initialisation."""
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)


# ---------------------------------------------------------------------------
# LSTM Forecaster
# ---------------------------------------------------------------------------

class LSTMForecaster:
    """Stacked LSTM for sequence-to-one time series forecasting.

    Architecture
    ------------
    - ``num_layers`` stacked LSTM cells
    - Each cell has forget, input, candidate, and output gates
    - The final hidden state of the top layer is passed through a linear
      projection to produce the scalar forecast.

    Training uses manual Backpropagation Through Time (BPTT) -- the entire
    sequence is unrolled and gradients are accumulated backward through all
    time steps.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the hidden state (default 32).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    dropout : float
        Dropout probability applied between LSTM layers (default 0.0).
    input_size : int, optional
        Number of input features.  Inferred from data during ``fit()``.
    random_state : int, optional
        Seed for weight initialisation.

    Examples
    --------
    >>> model = LSTMForecaster(hidden_size=32, num_layers=2, dropout=0.1)
    >>> model.fit(X_train, y_train, epochs=50, lr=0.001)
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
        input_size: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_size = input_size
        self.random_state = random_state

        # Weights will be initialised once input_size is known
        self._params: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def _init_params(self, input_size: int) -> None:
        """Initialise all LSTM weight matrices and biases."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.input_size = input_size
        H = self.hidden_size

        # Each of the 4 gates (f, i, g, o) has:
        #   W_xh : (input_size, hidden_size)
        #   W_hh : (hidden_size, hidden_size)
        #   b    : (1, hidden_size) or (hidden_size,)
        gate_label = ["f", "i", "g", "o"]

        for layer in range(self.num_layers):
            in_dim = input_size if layer == 0 else H
            prefix = f"l{layer}_"

            for gate in gate_label:
                self._params[prefix + f"W_xh_{gate}"] = _xavier_init((in_dim, H))
                self._params[prefix + f"W_hh_{gate}"] = _xavier_init((H, H))
                self._params[prefix + f"b_{gate}"] = np.zeros((1, H))

        # Output projection: hidden_size -> 1
        self._params["W_out"] = _xavier_init((H, 1))
        self._params["b_out"] = np.zeros((1, 1))

    # ------------------------------------------------------------------
    # Forward pass (training mode -- caches intermediates for BPTT)
    # ------------------------------------------------------------------

    def _forward(
        self, X: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Forward pass through the stacked LSTM.

        Parameters
        ----------
        X : np.ndarray, shape (batch, seq_len, input_size)
        training : bool
            If True, cache all intermediate values for backward pass.

        Returns
        -------
        outputs : np.ndarray, shape (batch, 1)
        cache : dict or None
            Intermediate values (only when training=True).
        """
        batch_size, seq_len, _ = X.shape
        H = self.hidden_size
        L = self.num_layers
        gate_label = ["f", "i", "g", "o"]

        # Initial hidden and cell states: zero for each layer
        h = [np.zeros((batch_size, H)) for _ in range(L)]
        c = [np.zeros((batch_size, H)) for _ in range(L)]

        cache: Dict = {
            "X": X,
            "h": [[] for _ in range(L)],   # h[l][t]  (batch, H)
            "c": [[] for _ in range(L)],   # c[l][t]  (batch, H)
            "gates": [[] for _ in range(L)],  # gates[l][t] = {f,i,g,o}
            "pre_act": [[] for _ in range(L)],  # pre-activation for each gate
            "h_before_dropout": [[] for _ in range(L - 1)],
            "dropout_mask": [[] for _ in range(L - 1)],
        } if training else None

        for t in range(seq_len):
            layer_input = X[:, t, :]  # (batch, input_size)

            for l in range(L):
                prefix = f"l{l}_"

                # Concatenate [layer_input, h_prev] or [x_t, h_{t-1}^l]
                h_prev = h[l]

                # ---- Compute gate pre-activations via manual matrix multiply ----
                # z_f = layer_input @ W_xh_f + h_prev @ W_hh_f + b_f
                pre_f = (
                    layer_input @ self._params[prefix + "W_xh_f"]
                    + h_prev @ self._params[prefix + "W_hh_f"]
                    + self._params[prefix + "b_f"]
                )
                pre_i = (
                    layer_input @ self._params[prefix + "W_xh_i"]
                    + h_prev @ self._params[prefix + "W_hh_i"]
                    + self._params[prefix + "b_i"]
                )
                pre_g = (
                    layer_input @ self._params[prefix + "W_xh_g"]
                    + h_prev @ self._params[prefix + "W_hh_g"]
                    + self._params[prefix + "b_g"]
                )
                pre_o = (
                    layer_input @ self._params[prefix + "W_xh_o"]
                    + h_prev @ self._params[prefix + "W_hh_o"]
                    + self._params[prefix + "b_o"]
                )

                # ---- Activations ----
                f = _sigmoid(pre_f)
                i = _sigmoid(pre_i)
                g = np.tanh(pre_g)
                o = _sigmoid(pre_o)

                # ---- Cell state update ----
                c[l] = f * c[l] + i * g

                # ---- Hidden state update ----
                h_new = o * np.tanh(c[l])

                if cache is not None:
                    cache["h"][l].append(h_new)
                    cache["c"][l].append(c[l])
                    cache["gates"][l].append({"f": f, "i": i, "g": g, "o": o})
                    cache["pre_act"][l].append(
                        {"f": pre_f, "i": pre_i, "g": pre_g, "o": pre_o}
                    )

                h[l] = h_new

                # Dropout between layers (not applied to output of last layer)
                if self.dropout > 0 and l < L - 1 and training:
                    keep_mask = (
                        np.random.rand(*h[l].shape) > self.dropout
                    ).astype(np.float64) / (1.0 - self.dropout)
                    if cache is not None:
                        cache["h_before_dropout"][l].append(h[l])
                        cache["dropout_mask"][l].append(keep_mask)
                    h[l] = h[l] * keep_mask

                layer_input = h[l]  # feed to next layer

        # ---- Output projection ----
        # Use the final hidden state of the top layer
        out = h[-1] @ self._params["W_out"] + self._params["b_out"]  # (batch, 1)

        return out, cache

    # ------------------------------------------------------------------
    # Backward pass (BPTT)
    # ------------------------------------------------------------------

    def _backward(
        self, cache: Dict, y: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Backpropagation Through Time for stacked LSTM.

        Parameters
        ----------
        cache : dict
            Intermediate values from forward pass.
        y : np.ndarray, shape (batch, 1)
            Target values.
        y_pred : np.ndarray, shape (batch, 1)
            Predicted values.

        Returns
        -------
        grads : dict
            Gradients for each parameter.
        """
        batch_size, seq_len, _ = cache["X"].shape
        H = self.hidden_size
        L = self.num_layers
        gate_label = ["f", "i", "g", "o"]

        grads: Dict[str, np.ndarray] = {}

        # ---- Gradient of MSE loss w.r.t. output ----
        d_out = (2.0 / batch_size) * (y_pred - y)  # (batch, 1)

        # ---- Output projection gradients ----
        h_last = cache["h"][-1][-1]  # (batch, H)
        grads["W_out"] = h_last.T @ d_out  # (H, 1)
        grads["b_out"] = np.sum(d_out, axis=0, keepdims=True)  # (1, 1)

        # ---- Gradient flowing into the top layer's last hidden state ----
        dh_next = [np.zeros((batch_size, H)) for _ in range(L)]
        dc_next = [np.zeros((batch_size, H)) for _ in range(L)]

        # d_out contributes to dh of top layer's last time step
        dh_next[-1] = d_out @ self._params["W_out"].T  # (batch, H)

        # ---- BPTT: iterate backward through time ----
        for t in reversed(range(seq_len)):
            for l in reversed(range(L)):
                prefix = f"l{l}_"

                # Recover cached values for this layer & time step
                f = cache["gates"][l][t]["f"]
                i = cache["gates"][l][t]["i"]
                g = cache["gates"][l][t]["g"]
                o = cache["gates"][l][t]["o"]

                c_t = cache["c"][l][t]
                c_prev = cache["c"][l][t - 1] if t > 0 else np.zeros((batch_size, H))

                # Input to this layer at time t
                if l == 0:
                    x_t = cache["X"][:, t, :]
                else:
                    # The input from the previous layer (after potential dropout)
                    if self.dropout > 0 and t < seq_len:
                        x_t = cache["h"][l - 1][t]
                        if t < len(cache.get("dropout_mask", [[] for _ in range(L - 1)])):
                            pass  # dropout handled in dh_prev
                    else:
                        x_t = cache["h"][l - 1][t] if t < len(cache["h"][l - 1]) else np.zeros((batch_size, H))

                # dh_next[l] already includes gradient from:
                #   - output layer (if l is top layer)
                #   - layer above (if l < top layer)
                #   - next time step (if t < seq_len-1)

                # ---- Output gate gradients ----
                tanh_c = np.tanh(c_t)
                do = dh_next[l] * tanh_c * _sigmoid_derivative(o)

                # ---- Cell state gradients ----
                dc = dh_next[l] * o * _tanh_derivative(tanh_c) + dc_next[l]

                # ---- Forget gate gradients ----
                df = dc * c_prev * _sigmoid_derivative(f)

                # ---- Input gate gradients ----
                di = dc * g * _sigmoid_derivative(i)

                # ---- Candidate gate gradients ----
                dg = dc * i * _tanh_derivative(g)

                # ---- Accumulate gradients for this layer's weights ----
                # Each gate contributes: dx = d_gate @ W_xh_gate^T
                #                       dh_prev = d_gate @ W_hh_gate^T
                #                       dW_xh = x^T @ d_gate
                #                       dW_hh = h_prev^T @ d_gate

                # Input to this LSTM cell
                if l == 0:
                    layer_input = cache["X"][:, t, :]
                else:
                    layer_input = cache["h"][l - 1][t]

                h_prev_l = cache["h"][l][t - 1] if t > 0 else np.zeros((batch_size, H))

                for gate_name, d_gate in zip(gate_label, [df, di, dg, do]):
                    w_xh_key = prefix + f"W_xh_{gate_name}"
                    w_hh_key = prefix + f"W_hh_{gate_name}"
                    b_key = prefix + f"b_{gate_name}"

                    grads.setdefault(w_xh_key, np.zeros_like(self._params[w_xh_key]))
                    grads.setdefault(w_hh_key, np.zeros_like(self._params[w_hh_key]))
                    grads.setdefault(b_key, np.zeros_like(self._params[b_key]))

                    grads[w_xh_key] += layer_input.T @ d_gate
                    grads[w_hh_key] += h_prev_l.T @ d_gate
                    grads[b_key] += np.sum(d_gate, axis=0, keepdims=True)

                # ---- Gradient to pass to the previous time step ----
                # dh_prev = sum over gates of (d_gate @ W_hh^T)
                dh_prev_time = np.zeros((batch_size, H))
                for gate_name, d_gate in zip(gate_label, [df, di, dg, do]):
                    dh_prev_time += d_gate @ self._params[prefix + f"W_hh_{gate_name}"].T

                dc_next[l] = dc * f  # cell state gradient to t-1
                dh_next[l] = dh_prev_time

                # ---- Gradient to pass to the layer below ----
                if l > 0:
                    dx_in = np.zeros((batch_size, H))
                    for gate_name, d_gate in zip(gate_label, [df, di, dg, do]):
                        dx_in += d_gate @ self._params[prefix + f"W_xh_{gate_name}"].T

                    # Handle dropout gradient
                    if self.dropout > 0 and l - 1 < len(cache.get("dropout_mask", [])):
                        if t < len(cache["dropout_mask"][l - 1]):
                            mask = cache["dropout_mask"][l - 1][t]
                            dx_in = dx_in * mask

                    dh_next[l - 1] += dx_in

        return grads

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
    ) -> "LSTMForecaster":
        """Train the LSTM forecaster.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, seq_len, n_features)
            Input sequences.
        y : np.ndarray, shape (n_samples,)
            Target values.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        verbose : bool
            Print loss every 10 epochs.

        Returns
        -------
        self
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.input_size is None:
            self._init_params(X.shape[2])

        self._train_losses: List[float] = []

        for epoch in range(epochs):
            # ---- Forward ----
            y_pred, cache = self._forward(X, training=True)

            # ---- Loss ----
            loss = np.mean((y_pred - y) ** 2)
            self._train_losses.append(float(loss))

            # ---- Backward ----
            grads = self._backward(cache, y, y_pred)

            # ---- Update ----
            for key in self._params:
                if key in grads:
                    self._params[key] -= lr * grads[key]

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1:4d}/{epochs}  loss = {loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate forecasts for input sequences.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, seq_len, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted target values.
        """
        y_pred, _ = self._forward(X, training=False)
        return y_pred.ravel()

    def count_params(self) -> int:
        """Return the total number of trainable parameters."""
        return int(sum(p.size for p in self._params.values()))
