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


def profit_factor(returns):
    returns = np.asarray(returns, dtype=float)
    pos = returns[returns > 0].sum()
    neg = returns[returns < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))


def trade_stats(signals, close):
    """Conta trades (corridas de posição não-zero) e calcula win_rate por P&L acumulado.

    Signature: trade_stats(signals, close) -> (n_trades: int, win_rate: float)
    Um trade = sequência contígua de barras com mesma posição não-zero (corrida).
    P&L do trade = produto acumulado de (1 + bar_return) - 1.
    """
    signals = pd.Series(signals).reset_index(drop=True).astype(float)
    rets = pd.Series(bar_returns(signals, close))
    eff = signals.shift(1).fillna(0.0)            # posição efetiva por barra
    run_id = (eff != eff.shift(1)).cumsum()       # id de corrida (valor constante)
    out = pd.DataFrame({"eff": eff, "ret": rets, "run": run_id})
    trades = out[out["eff"] != 0].groupby("run")
    n_trades = trades.ngroups
    if n_trades == 0:
        return 0, 0.0
    pnl = trades["ret"].apply(lambda r: (1.0 + r).prod() - 1.0)
    win_rate = float((pnl > 0).mean())
    return int(n_trades), win_rate
