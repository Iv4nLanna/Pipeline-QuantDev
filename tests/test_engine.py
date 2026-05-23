import numpy as np
import pandas as pd
import pytest
from validation.engine import evaluate_grid
from strategies.donchian import donchian


def _trend_df(n=80):
    p = pd.Series(np.linspace(100, 200, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    arr = p.values
    return pd.DataFrame({"open": arr, "high": arr, "low": arr, "close": arr, "volume": 1.0}, index=idx)


def test_evaluate_grid_shape_e_colunas():
    df = _trend_df()
    grid = {"entry_lookback": [3, 4], "exit_lookback": [2]}
    out = evaluate_grid(donchian, df, grid)
    assert list(out.index.names) == ["entry_lookback", "exit_lookback"]
    assert len(out) == 2
    assert {"profit_factor", "win_rate", "n_trades", "n_bars"}.issubset(out.columns)
    assert (out["n_bars"] == len(df)).all()


def test_evaluate_grid_param_grid_vazio():
    with pytest.raises(ValueError):
        evaluate_grid(donchian, _trend_df(), {})


def test_evaluate_grid_pf_consistente_com_metrics():
    from validation.metrics import bar_returns, profit_factor

    df = _trend_df()
    out = evaluate_grid(donchian, df, {"entry_lookback": [3], "exit_lookback": [2]})
    sig = donchian(df, entry_lookback=3, exit_lookback=2)
    pf_esperado = profit_factor(bar_returns(sig, df["close"]))
    assert out["profit_factor"].iloc[0] == pf_esperado
