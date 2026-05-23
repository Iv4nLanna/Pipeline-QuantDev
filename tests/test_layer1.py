import os
import pandas as pd
import config
from validation.layer1 import in_sample_excellence


def _df(prices):
    p = pd.Series(prices, dtype=float)
    idx = pd.date_range("2020-01-01", periods=len(p), freq="1h")
    arr = p.values
    return pd.DataFrame({"open": arr, "high": arr, "low": arr, "close": arr, "volume": 1.0}, index=idx)


def test_in_sample_excellence_aprovado(tmp_path, monkeypatch):
    def fake(df, a, b):
        return pd.Series(1.0, index=df.index)

    monkeypatch.setattr(config, "REDFLAG_MAX_PF", 1000.0)
    monkeypatch.setattr(config, "REDFLAG_MIN_TRADES", 1)
    monkeypatch.setattr(config, "REDFLAG_WIN_RATE", 1.01)

    df = _df([100, 101, 102, 101.5, 103, 104])
    grid = {"a": [1, 2], "b": [3, 4]}
    r = in_sample_excellence(fake, df, grid, results_dir=str(tmp_path))

    assert r["camada"] == "in_sample_excellence"
    assert r["status"] == "aprovado"
    assert {"best_param", "profit_factor", "win_rate", "n_trades", "n_bars", "heatmap"} <= set(r["metricas"])
    assert os.path.exists(r["metricas"]["heatmap"])


def test_in_sample_excellence_reprovado_red_flag_min_trades(tmp_path):
    # poucas barras -> 1 trade só -> red-flag de poder estatístico (n_trades < 30)
    def fake(df, a, b):
        return pd.Series(1.0, index=df.index)

    df = _df([100, 101, 102, 103, 104])
    grid = {"a": [1, 2], "b": [3, 4]}
    r = in_sample_excellence(fake, df, grid, results_dir=str(tmp_path))
    assert r["status"] == "reprovado"
    assert "n_trades" in r["motivo"]


def test_in_sample_excellence_aprova_pf_baixo_sem_red_flags(tmp_path, monkeypatch):
    """Passo 1 não rejeita por magnitude de PF: PF < 1.0 sem red-flags = aprovado.
    Significância é tarefa do Passo 2 (permutation_test_is).
    """
    monkeypatch.setattr(config, "REDFLAG_MIN_TRADES", 1)

    # estratégia perdedora -> PF < 1.0 mas sem red-flag de overfit
    def fake_loser(df, a, b):
        return pd.Series(1.0, index=df.index)

    df = _df([100, 99, 98, 97])  # cai -> PF < 1.0
    grid = {"a": [1, 2], "b": [3, 4]}
    r = in_sample_excellence(fake_loser, df, grid, results_dir=str(tmp_path))
    assert r["status"] == "aprovado"
    assert r["metricas"]["profit_factor"] < 1.0
