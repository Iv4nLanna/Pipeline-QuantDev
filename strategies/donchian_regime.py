import pandas as pd

import config
from data.layer0 import detect_regime
from strategies.donchian import donchian


def donchian_filtered_regime(df, entry_lookback, exit_lookback, ma_period=None):
    """Donchian breakout, mas só atua em regime de tendência.

    Hipótese econômica: breakout em cripto só tem edge quando o mercado
    está em tendência (bull ou bear). Em regime lateral o filtro força flat.

    O regime usa a média móvel de `ma_period` (default `config.REGIME_MA_PERIOD`)
    sobre o close: regime[t] = "bull" se close[t] > MA[t] e MA está subindo;
    "bear" se inverso; "lateral" caso contrário; NaN no aquecimento da MA.

    Anti-lookahead: regime[t] e donchian_sig[t] usam só info até a barra t;
    a defasagem de execução de 1 barra é aplicada depois em metrics.bar_returns
    (signals.shift(1)), igual ao Donchian puro.
    """
    if ma_period is None:
        ma_period = config.REGIME_MA_PERIOD
    base = donchian(df, entry_lookback=entry_lookback, exit_lookback=exit_lookback)
    regime = detect_regime(df, ma_period=ma_period)["regime"]
    em_tendencia = regime.isin(["bull", "bear"])   # False quando NaN ou "lateral"
    return base.where(em_tendencia, 0.0)
