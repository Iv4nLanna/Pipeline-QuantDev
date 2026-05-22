# validation/metrics.py
import numpy as np
import pandas as pd


def bar_returns(signals, close):
    """Retornos barra-a-barra com defasagem de execução de 1 barra (anti-lookahead).

    posição efetiva na barra t = signals[t-1]. long: close[t]/close[t-1]-1;
    short: close[t-1]/close[t]-1; flat: 0.
    """
    signals = pd.Series(signals).reset_index(drop=True).astype(float)
    close = pd.Series(close).reset_index(drop=True).astype(float)
    pos = signals.shift(1).fillna(0.0)
    prev = close.shift(1)
    long_ret = close / prev - 1.0
    short_ret = prev / close - 1.0
    ret = pd.Series(0.0, index=close.index)
    ret[pos == 1] = long_ret[pos == 1]
    ret[pos == -1] = short_ret[pos == -1]
    return ret.fillna(0.0).to_numpy()
