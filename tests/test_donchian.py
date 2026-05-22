import numpy as np
import pandas as pd
from strategies.donchian import donchian


def _df(prices):
    p = pd.Series(prices, dtype=float)
    idx = pd.date_range("2020-01-01", periods=len(p), freq="1h")
    arr = p.values
    return pd.DataFrame({"open": arr, "high": arr, "low": arr, "close": arr, "volume": 1.0}, index=idx)


def test_donchian_entrada_long_e_saida_flat():
    # rompe a máxima de 3 barras em t=4 (long); rompe o canal de saída (2) em t=7 (flat)
    df = _df([100, 100, 100, 100, 105, 106, 107, 105.5, 104, 103])
    sig = donchian(df, entry_lookback=3, exit_lookback=2)
    assert (sig.iloc[:4] == 0).all()     # sem canal / sem rompimento
    assert sig.iloc[4] == 1              # entrada long
    assert sig.iloc[5] == 1 and sig.iloc[6] == 1
    assert sig.iloc[7] == 0              # saída pelo canal de exit_lookback


def test_donchian_sem_lookahead():
    df = _df([100, 100, 100, 100, 105, 106, 107, 105.5, 104, 103])
    sig_full = donchian(df, entry_lookback=3, exit_lookback=2)
    df2 = df.copy()
    df2.iloc[9, df2.columns.get_loc("close")] = 999  # altera só a última barra
    df2.iloc[9, df2.columns.get_loc("high")] = 999
    sig2 = donchian(df2, entry_lookback=3, exit_lookback=2)
    # mudar o futuro (t=9) não pode alterar os sinais anteriores
    pd.testing.assert_series_equal(sig_full.iloc[:9], sig2.iloc[:9])


def test_donchian_valida_colunas():
    import pytest
    bad = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        donchian(bad, entry_lookback=3, exit_lookback=2)
