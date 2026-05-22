import os
import time

import ccxt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from utils.contract import make_result, save_result


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


def _plot_regime(df, ma_period, results_dir):
    from datetime import datetime
    os.makedirs(results_dir, exist_ok=True)
    ma = df["close"].rolling(ma_period).mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["close"], lw=0.8, label="close")
    ax.plot(df.index, ma, lw=1.0, label=f"MA{ma_period}")
    cores = {"bull": "#c8f7c5", "bear": "#f7c5c5", "lateral": "#eeeeee"}
    for reg, cor in cores.items():
        mask = df["regime"] == reg
        ax.fill_between(df.index, df["close"].min(), df["close"].max(),
                        where=mask, color=cor, alpha=0.3, step="mid")
    ax.legend(); ax.set_title("Camada 0 — preço, MA e regime")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(results_dir, f"regime_{ts}.png")
    fig.savefig(caminho, dpi=110, bbox_inches="tight"); plt.close(fig)
    return caminho


def run_layer0(ticker, timeframe, start, end, ma_period=200,
               results_dir="results", cache_dir="data/cache", exchange_name="binance"):
    bruto = fetch_data(ticker, timeframe, start, end,
                       cache_dir=cache_dir, exchange_name=exchange_name)
    n_bruto = len(bruto)
    limpo = clean_data(bruto)
    com_regime = detect_regime(limpo, ma_period=ma_period)

    pct_removido = 0.0 if n_bruto == 0 else round(100 * (n_bruto - len(limpo)) / n_bruto, 3)
    regimes = com_regime["regime"].dropna()
    dist = {} if regimes.empty else (regimes.value_counts(normalize=True) * 100).round(2).to_dict()
    ret = limpo["close"].pct_change().dropna()
    metricas = {
        "n_candles": int(len(limpo)),
        "intervalo": [str(limpo.index.min()), str(limpo.index.max())] if len(limpo) else [],
        "pct_removido_limpeza": pct_removido,
        "distribuicao_regimes": dist,
        "retorno_medio": round(float(ret.mean()), 8) if len(ret) else 0.0,
        "retorno_std": round(float(ret.std()), 8) if len(ret) else 0.0,
        "skew": round(float(skew(ret)), 4) if len(ret) > 2 else 0.0,
        "kurtosis": round(float(kurtosis(ret)), 4) if len(ret) > 2 else 0.0,
    }

    if len(limpo) >= ma_period:
        grafico = _plot_regime(com_regime, ma_period, results_dir)
        metricas["grafico_regime"] = grafico
        status, motivo = "aprovado", f"{len(limpo)} candles limpos, regimes detectados."
        proximo = "Avançar para Camada 1 (validação estatística)."
    else:
        status = "reprovado"
        motivo = f"Apenas {len(limpo)} candles após limpeza; mínimo {ma_period}."
        proximo = "Ampliar intervalo ou revisar fonte de dados."

    resultado = make_result("camada0_dados", status, metricas, motivo, proximo)
    save_result(resultado, results_dir=results_dir)
    return resultado
