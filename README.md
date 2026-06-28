# EconNet

**Deep Learning for Economic Time Series Forecasting**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

EconNet is a **pure NumPy** deep learning toolkit for economic time series forecasting. All models are implemented from scratch using manual forward/backward passes -- **zero deep learning framework dependencies** (no PyTorch, no TensorFlow).

## Features

- **LSTM** -- Stacked Long Short-Term Memory with manual BPTT (Backpropagation Through Time)
- **TCN** -- Temporal Convolutional Network with dilated causal convolutions and residual blocks
- **Transformer** -- Time-series Transformer with sinusoidal positional encoding and multi-head self-attention
- **N-BEATS** -- Neural Basis Expansion Analysis for interpretable time series forecasting

All models share a consistent `fit(X, y)` / `predict(X)` API.

## Installation

```bash
pip install econnet
```

Or install from source:

```bash
git clone https://github.com/econnet/econnet.git
cd econnet
pip install -e .
```

## Quick Start

```python
import numpy as np
from econnet.datasets import generate_economic_data
from econnet.models import LSTMForecaster
from econnet.utils import train_test_split, StandardScaler

# Generate synthetic economic data
X, y = generate_economic_data(n=2000)

# Split and normalize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train an LSTM forecaster
model = LSTMForecaster(hidden_size=32, num_layers=2, dropout=0.1)
model.fit(X_train_s, y_train, epochs=50, lr=0.001)

# Predict
y_pred = model.predict(X_test_s)
```

## Model Architectures

### LSTM (`LSTMForecaster`)
- Configurable `hidden_size`, `num_layers`, `dropout`
- Sequence-to-one prediction
- Pure NumPy implementation with manual BPTT

### TCN (`TCNForecaster`)
- Dilated causal convolutions (1, 2, 4, 8, ...)
- Residual blocks with skip connections
- Manual convolution via nested loops

### Transformer (`TransformerForecaster`)
- Sinusoidal positional encoding
- Multi-head self-attention: `Softmax(QK^T/sqrt(d_k)) V`
- Feed-forward layers with ReLU and layer normalization

### N-BEATS (`NBEATSForecaster`)
- Block architecture with basis expansion (trend + seasonality)
- Backcast/forecast projection to basis functions
- Stacked blocks with doubly residual connections

## Datasets

```python
from econnet.datasets import (
    generate_arma,           # ARMA process generator
    generate_economic_data,  # GDP + inflation + unemployment
    generate_financial_data, # Stock returns with volatility clustering
)
```

## Example

Run the demo script to compare all four models:

```bash
python examples/demo.py
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0

## License

MIT License -- see [LICENSE](LICENSE) for details.

## Citation

If you use EconNet in your research, please cite:

```bibtex
@software{econnet2024,
  author = {EconNet Contributors},
  title = {EconNet: Deep Learning for Economic Time Series Forecasting},
  year = {2024},
  url = {https://github.com/econnet/econnet},
}
```
