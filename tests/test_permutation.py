import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew, kurtosis
from validation.permutation import get_permutation


def _ar1(n, phi=0.6, sigma=0.01, seed=0):
    """Série de log-retornos AR(1) com autocorr forte; preço positivo via cumsum."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + eps[t]
    close = 100.0 * np.exp(np.cumsum(r))
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1.0}, index=idx
    )


def test_get_permutation_preserva_momentos():
    df = _ar1(n=5000)
    perm = get_permutation(df, seed=42)
    r_real = np.log(df["close"]).diff().dropna().values
    r_perm = np.log(perm["close"]).diff().dropna().values
    assert abs(r_perm.mean() - r_real.mean()) / max(abs(r_real.mean()), 1e-9) < 0.05
    assert abs(r_perm.std() - r_real.std()) / r_real.std() < 0.05
    assert abs(skew(r_perm) - skew(r_real)) < 0.1
    assert abs(kurtosis(r_perm) - kurtosis(r_real)) < 0.5


def test_get_permutation_destroi_autocorrelacao():
    df = _ar1(n=5000)
    perm = get_permutation(df, seed=42)
    r_real = pd.Series(np.log(df["close"]).diff().dropna().values)
    r_perm = pd.Series(np.log(perm["close"]).diff().dropna().values)
    assert abs(r_real.autocorr(lag=1)) > 0.5
    assert abs(r_perm.autocorr(lag=1)) < 0.05


def test_get_permutation_reprodutivel_com_seed():
    df = _ar1(n=200)
    a = get_permutation(df, seed=7)
    b = get_permutation(df, seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = get_permutation(df, seed=8)
    assert not a["close"].equals(c["close"])


def test_get_permutation_preserva_ohlc_via_multiplier():
    df = _ar1(n=500)
    perm = get_permutation(df, seed=3)
    np.testing.assert_allclose(perm["open"] / perm["close"],
                               df["open"] / df["close"], rtol=1e-9)
    np.testing.assert_allclose(perm["high"] / perm["close"],
                               df["high"] / df["close"], rtol=1e-9)
    np.testing.assert_allclose(perm["low"] / perm["close"],
                               df["low"] / df["close"], rtol=1e-9)
    np.testing.assert_array_equal(perm["volume"].values, df["volume"].values)
