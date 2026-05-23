import pytest
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


from data.layer0 import detect_regime


def test_detect_regime_classifica_tendencias():
    n = 250
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    subida = pd.DataFrame({"close": np.linspace(100, 300, n)}, index=idx)
    out_bull = detect_regime(subida, ma_period=200)
    assert (out_bull["regime"].dropna() == "bull").mean() > 0.8

    descida = pd.DataFrame({"close": np.linspace(300, 100, n)}, index=idx)
    out_bear = detect_regime(descida, ma_period=200)
    assert (out_bear["regime"].dropna() == "bear").mean() > 0.8


def test_detect_regime_adiciona_coluna_e_nan_no_aquecimento():
    idx = pd.date_range("2020-01-01", periods=250, freq="1h")
    df = pd.DataFrame({"close": np.linspace(100, 300, 250)}, index=idx)
    out = detect_regime(df, ma_period=200)
    assert "regime" in out.columns
    assert out["regime"].iloc[:199].isna().all()


from data import layer0


def test_fetch_data_le_do_cache_sem_rede(tmp_path, monkeypatch):
    # prepara um parquet de cache cobrindo o intervalo pedido
    idx = pd.date_range("2020-01-01", periods=48, freq="1h", tz="UTC")
    cache_df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx
    )
    cache_df.index.name = "datetime"
    cache_path = tmp_path / "btcusdt_1h.parquet"
    cache_df.to_parquet(cache_path)

    # se _download_ccxt for chamado, o teste falha (não deve haver rede)
    def _boom(*a, **k):
        raise AssertionError("não deveria baixar da rede quando o cache cobre o intervalo")

    monkeypatch.setattr(layer0, "_download_ccxt", _boom)

    out = layer0.fetch_data(
        "BTC/USDT", "1h", "2020-01-01", "2020-01-02",
        use_cache=True, cache_dir=str(tmp_path),
    )
    assert len(out) >= 24
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_run_layer0_retorna_dict_padronizado(tmp_path, monkeypatch):
    idx = pd.date_range("2020-01-01", periods=300, freq="1h", tz="UTC")
    fake = pd.DataFrame(
        {"open": np.linspace(100, 200, 300), "high": np.linspace(100, 200, 300),
         "low": np.linspace(100, 200, 300), "close": np.linspace(100, 200, 300),
         "volume": np.full(300, 5.0)}, index=idx)
    fake.index.name = "datetime"

    monkeypatch.setattr(layer0, "fetch_data", lambda *a, **k: fake)

    r = layer0.run_layer0(
        "BTC/USDT", "1h", "2020-01-01", "2020-01-13",
        ma_period=200, results_dir=str(tmp_path), cache_dir=str(tmp_path),
    )
    assert set(r.keys()) == {"camada", "status", "metricas", "motivo", "proximo_passo"}
    assert r["status"] in {"aprovado", "reprovado", "revisar"}
    assert r["camada"] == "camada0_dados"
    assert "distribuicao_regimes" in r["metricas"]


def test_load_clean_ohlc_reusa_fetch_e_clean(tmp_path, monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC")
    raw = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 1.0, 1.0, 1.0, 1.0],
         "volume": [1.0, 1.0, 0.0, 1.0, 1.0]},
        index=idx,
    )
    monkeypatch.setattr(layer0, "fetch_data", lambda *a, **k: raw)
    out = layer0.load_clean_ohlc("BTC/USDT", "1h", "2020-01-01", "2020-01-02",
                                 cache_dir=str(tmp_path))
    assert (out["volume"] > 0).all()
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


@pytest.mark.network
def test_fetch_data_integracao_binance(tmp_path):
    df = layer0.fetch_data(
        "BTC/USDT", "1h", "2021-01-01", "2021-01-03",
        use_cache=False, cache_dir=str(tmp_path),
    )
    assert not df.empty, "fetch_data deve retornar DataFrame não-vazio"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None, "índice deve ser tz-aware (UTC)"
    assert df.index.is_monotonic_increasing, "índice deve ser monotônico crescente"
    # 2 dias de candles 1h: esperado ~48; aceita >= 40 (margem para fins de período)
    assert len(df) >= 40, f"esperado >= 40 candles; obteve {len(df)}"
