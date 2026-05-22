import numpy as np
import pandas as pd


def donchian(df, entry_lookback, exit_lookback):
    """Donchian breakout long+short (estilo Turtle) -> Series de {-1, 0, 1}.

    Canais usam apenas barras anteriores (shift(1)); a defasagem de execução
    é aplicada depois, em metrics.bar_returns.
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"donchian requer a coluna '{col}'")
    if df.empty:
        raise ValueError("donchian requer DataFrame não-vazio")

    entry_hi = df["high"].rolling(entry_lookback).max().shift(1).to_numpy()
    entry_lo = df["low"].rolling(entry_lookback).min().shift(1).to_numpy()
    exit_hi = df["high"].rolling(exit_lookback).max().shift(1).to_numpy()
    exit_lo = df["low"].rolling(exit_lookback).min().shift(1).to_numpy()
    c = df["close"].to_numpy()

    n = len(df)
    sig = np.zeros(n)
    pos = 0
    for t in range(n):
        if np.isnan(entry_hi[t]):
            sig[t] = 0
            continue
        if pos == 0:
            if c[t] > entry_hi[t]:
                pos = 1
            elif c[t] < entry_lo[t]:
                pos = -1
        elif pos == 1:
            if c[t] < entry_lo[t]:        # rompe canal de entrada oposto -> inverte short
                pos = -1
            elif c[t] < exit_lo[t]:        # rompe canal de saída -> flat
                pos = 0
        elif pos == -1:
            if c[t] > entry_hi[t]:         # inverte long
                pos = 1
            elif c[t] > exit_hi[t]:        # saída -> flat
                pos = 0
        sig[t] = pos
    return pd.Series(sig, index=df.index, dtype=float)
