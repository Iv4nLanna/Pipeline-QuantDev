# tests/test_metrics.py
import numpy as np
import pandas as pd
from validation.metrics import bar_returns


def test_bar_returns_long_e_short_com_shift():
    close = pd.Series([100.0, 110.0, 99.0])
    signals = pd.Series([1, 1, -1])
    # pos = signals.shift(1) = [0, 1, 1] -> barra0 flat; barra1 long; barra2 long
    r = bar_returns(signals, close)
    assert r[0] == 0.0
    assert abs(r[1] - 0.10) < 1e-9          # long: 110/100 - 1
    assert abs(r[2] - (99 / 110 - 1)) < 1e-9  # long: 99/110 - 1


def test_bar_returns_short():
    close = pd.Series([100.0, 90.0])
    signals = pd.Series([-1, -1])
    # pos = [0, -1] -> barra1 short: 100/90 - 1
    r = bar_returns(signals, close)
    assert r[0] == 0.0
    assert abs(r[1] - (100 / 90 - 1)) < 1e-9
