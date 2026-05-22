import numpy as np
import pandas as pd
import pytest
from validation.engine import evaluate_grid
from strategies.donchian import donchian


def _trend_df(n=80):
    p = pd.Series(np.linspace(100, 200, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=idx)


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
