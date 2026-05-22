import pandas as pd
import numpy as np
from data.layer0 import clean_data


def _df(idx, close, volume):
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": volume},
        index=pd.DatetimeIndex(idx, name="datetime"),
    )


def test_clean_data_remove_duplicatas_e_volume_zero():
    idx = ["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
    df = _df(idx, close=[100, 100, 101, 102], volume=[10, 10, 0, 5])
    out = clean_data(df)
    # duplicata removida e candle de volume zero removido -> 2 linhas
    assert len(out) == 2
    assert out.index.is_monotonic_increasing
    assert (out["volume"] > 0).all()


def test_clean_data_remove_gap_anomalo():
    idx = pd.date_range("2020-01-01", periods=3, freq="1h")
    # segundo candle dobra de preço (+100%) -> anômalo, removido
    df = _df(idx, close=[100, 200, 201], volume=[5, 5, 5])
    out = clean_data(df, gap_threshold=0.5)
    assert 200 not in out["close"].values
