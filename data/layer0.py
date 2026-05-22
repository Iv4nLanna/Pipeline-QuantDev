import os
import time

import numpy as np
import pandas as pd


def clean_data(df, gap_threshold=0.5):
    if df.empty or "close" not in df.columns:
        raise ValueError("clean_data requer DataFrame não-vazio com coluna 'close'")
    out = df[~df.index.duplicated(keep="first")].copy()
    out = out.sort_index()
    out = out[out["volume"] > 0]
    ret = out["close"].pct_change().abs()
    out = out[(ret <= gap_threshold) | ret.isna()]
    return out


def detect_regime(df, ma_period=200):
    if df.empty or "close" not in df.columns:
        raise ValueError("detect_regime requer DataFrame não-vazio com coluna 'close'")
    out = df.copy()
    ma = out["close"].rolling(ma_period).mean()
    ma_slope = ma.diff()
    regime = pd.Series(np.nan, index=out.index, dtype="object")
    bull = (out["close"] > ma) & (ma_slope > 0)
    bear = (out["close"] < ma) & (ma_slope < 0)
    valido = ma.notna()
    regime[valido] = "lateral"
    regime[valido & bull] = "bull"
    regime[valido & bear] = "bear"
    out["regime"] = regime
    return out
