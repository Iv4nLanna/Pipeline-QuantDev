import os
import time

import ccxt
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


def _download_ccxt(ticker, timeframe, start_ms, end_ms, exchange_name="binance"):
    ex = getattr(ccxt, exchange_name)({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = start_ms
    linhas = []
    while since < end_ms:
        for tentativa in range(3):
            try:
                lote = ex.fetch_ohlcv(ticker, timeframe, since=since, limit=1000)
                break
            except ccxt.NetworkError:
                if tentativa == 2:
                    raise
                time.sleep(2 ** tentativa)
        if not lote:
            break
        linhas.extend(lote)
        since = lote[-1][0] + tf_ms
        if len(lote) < 1000:
            break
    df = pd.DataFrame(linhas, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns="ts")
    return df[df.index <= pd.to_datetime(end_ms, unit="ms", utc=True)]


def fetch_data(ticker, timeframe, start, end, use_cache=True,
               cache_dir="data/cache", exchange_name="binance"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "btcusdt_1h.parquet")
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)

    if use_cache and os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        if cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            return cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts),
                              ["open", "high", "low", "close", "volume"]]

    df = _download_ccxt(ticker, timeframe, int(start_ts.value // 1_000_000),
                        int(end_ts.value // 1_000_000), exchange_name)
    if use_cache:
        df.to_parquet(cache_path)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts),
                  ["open", "high", "low", "close", "volume"]]
