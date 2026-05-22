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
