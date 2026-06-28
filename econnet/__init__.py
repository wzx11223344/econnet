"""
EconNet: Deep Learning for Economic Time Series Forecasting.

A pure NumPy deep learning toolkit for economic time series forecasting,
featuring LSTM, TCN, Transformer, and N-BEATS architectures with
zero external deep learning framework dependencies.
"""

__version__ = "0.1.0"
__author__ = "EconNet Contributors"
__license__ = "MIT"

from econnet import models
from econnet import utils
from econnet import datasets

__all__ = ["models", "utils", "datasets"]
