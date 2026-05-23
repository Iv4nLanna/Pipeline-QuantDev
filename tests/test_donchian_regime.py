import numpy as np
import pandas as pd
import pytest

from strategies.donchian import donchian
from strategies.donchian_regime import donchian_filtered_regime


def _df(prices):
    p = pd.Series(prices, dtype=float)
    idx = pd.date_range("2020-01-01", periods=len(p), freq="1h")
    arr = p.values
    return pd.DataFrame({"open": arr, "high": arr, "low": arr, "close": arr,
                         "volume": 1.0}, index=idx)


def test_flat_no_warmup_da_ma():
    # MA precisa ma_period barras para ficar definida; antes disso regime=NaN -> sinal=0
    df = _df([100, 102, 104, 103, 110, 112, 115, 113, 120, 122])
    sig = donchian_filtered_regime(df, entry_lookback=3, exit_lookback=2, ma_period=5)
    # primeiras 4 barras: MA(5) ainda é NaN -> 0
    assert (sig.iloc[:4] == 0).all()


def test_passa_sinal_em_regime_bull():
    # tendência clara de alta -> regime "bull" após warmup; donchian gera +1 no rompimento
    n = 60
    prices = np.linspace(100, 200, n)               # subida monotônica
    df = _df(prices)
    sig_base = donchian(df, entry_lookback=5, exit_lookback=3)
    sig_filtrado = donchian_filtered_regime(df, entry_lookback=5, exit_lookback=3,
                                            ma_period=10)
    # depois do warmup, em bull, sinal filtrado == sinal base (todos +1 após rompimento)
    pos_long_base = sig_base[sig_base == 1].index
    pos_long_filtrado = sig_filtrado[sig_filtrado == 1].index
    assert len(pos_long_filtrado) >= 1
    assert pos_long_filtrado.isin(pos_long_base).all()


def test_zera_sinal_em_regime_lateral():
    # ruído ao redor de uma média estável -> regime "lateral" -> sinal forçado a 0
    n = 80
    rng = np.random.default_rng(0)
    prices = 100.0 + rng.normal(0, 0.5, n).cumsum() * 0.1   # passeio quase plano
    # garante média não-derivante: re-centra
    prices = prices - (prices.mean() - 100.0)
    df = _df(prices)
    sig_base = donchian(df, entry_lookback=5, exit_lookback=3)
    sig_filtrado = donchian_filtered_regime(df, entry_lookback=5, exit_lookback=3,
                                            ma_period=10)
    # em barras onde o base tinha sinal não-zero, o filtrado deve forçar 0
    # (verificação: pelo menos algumas dessas barras foram zeradas)
    barras_com_sinal_base = (sig_base != 0).sum()
    barras_zeradas = ((sig_base != 0) & (sig_filtrado == 0)).sum()
    assert barras_com_sinal_base > 0, "teste mal calibrado: base não gerou sinal"
    assert barras_zeradas > 0, "filtro deveria ter zerado pelo menos algumas barras"


def test_usa_ma_period_do_config_por_default():
    # quando ma_period não é passado, deve usar config.REGIME_MA_PERIOD (200)
    import config
    n = 250
    df = _df(np.linspace(100, 200, n))
    sig = donchian_filtered_regime(df, entry_lookback=5, exit_lookback=3)
    # com ma_period=200 default, as primeiras 199 barras têm regime NaN -> sig=0
    assert (sig.iloc[:199] == 0).all()
    # depois do warmup, em bull, espera ao menos uma posição long
    assert (sig.iloc[199:] == 1).any()


def test_sem_lookahead():
    """Alterar o futuro não pode alterar sinais passados."""
    df = _df(np.linspace(100, 200, 50))
    sig_full = donchian_filtered_regime(df, entry_lookback=5, exit_lookback=3,
                                        ma_period=10)
    df2 = df.copy()
    df2.iloc[49, df2.columns.get_loc("close")] = 999.0
    df2.iloc[49, df2.columns.get_loc("high")] = 999.0
    sig2 = donchian_filtered_regime(df2, entry_lookback=5, exit_lookback=3,
                                    ma_period=10)
    pd.testing.assert_series_equal(sig_full.iloc[:49], sig2.iloc[:49])


def test_valida_colunas():
    bad = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        donchian_filtered_regime(bad, entry_lookback=3, exit_lookback=2, ma_period=5)
