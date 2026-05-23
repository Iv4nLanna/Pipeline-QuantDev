import os
import numpy as np
import pandas as pd
import pytest
import config
from validation.layer1 import in_sample_excellence, permutation_test_is
from strategies.donchian import donchian
from validation.engine import evaluate_grid


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


# ---------- Passo 2: permutation_test_is ----------

def _df_tendencia(n=500, drift=0.001, sigma=0.005, seed=0):
    """Série OHLC com tendência forte + ruído gaussiano (seed fixa = reprodutível)."""
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(seed)
    r = drift + rng.normal(0, sigma, n)
    close = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1.0}, index=idx)


def _best_param_from_grid(grid_df):
    idx_max = grid_df["profit_factor"].idxmax()
    tup = idx_max if isinstance(idx_max, tuple) else (idx_max,)
    return {k: int(v) for k, v in zip(grid_df.index.names, tup)}


def test_permutation_test_is_estrutura_e_gates(tmp_path):
    df = _df_tendencia(n=400)
    grid = {"entry_lookback": [10, 20], "exit_lookback": [5, 10]}
    grid_real = evaluate_grid(donchian, df, grid)
    best_param = _best_param_from_grid(grid_real)
    pf_real = float(grid_real["profit_factor"].max())

    r = permutation_test_is(donchian, df, grid, best_param=best_param,
                            pf_real=pf_real, n_permutations=30, seed_base=0,
                            results_dir=str(tmp_path))
    assert r["camada"] == "permutation_test_is"
    assert r["status"] in {"aprovado", "reprovado"}
    assert 0.0 <= r["metricas"]["p_value"] <= 1.0
    assert r["metricas"]["n_permutations"] == 30
    assert r["metricas"]["profit_factor_real"] == pf_real
    assert r["metricas"]["best_param"] == best_param


def test_permutation_test_is_p_value_calculo_deterministico(tmp_path, monkeypatch):
    """Mock de evaluate_grid devolve PFs conhecidos: p_value = (count+1)/(n+1)."""
    df = _df_tendencia(n=100)
    pfs_perm_controlados = iter([0.5, 1.0, 1.5, 2.0, 2.5])  # 3 (>=1.5) extremos
    pf_real = 1.5
    # p_value = (3+1)/(5+1) = 4/6

    def fake_strategy(df, **p):
        return pd.Series(0.0, index=df.index)

    from validation import layer1 as _l1

    def fake_evaluate_grid(strategy_func, df, grid):
        pf = next(pfs_perm_controlados)
        return pd.DataFrame(
            {"profit_factor": [pf], "win_rate": [0.5], "n_trades": [10], "n_bars": [len(df)]},
            index=pd.MultiIndex.from_tuples([(1, 1)], names=["a", "b"]))

    monkeypatch.setattr(_l1, "evaluate_grid", fake_evaluate_grid)

    r = permutation_test_is(fake_strategy, df, {"a": [1], "b": [1]},
                            best_param={"a": 1, "b": 1}, pf_real=pf_real,
                            n_permutations=5, seed_base=0, results_dir=str(tmp_path))
    assert abs(r["metricas"]["p_value"] - 4 / 6) < 1e-9


def test_permutation_test_is_salva_array_e_histograma(tmp_path):
    df = _df_tendencia(n=200)
    grid = {"entry_lookback": [10], "exit_lookback": [5]}
    grid_real = evaluate_grid(donchian, df, grid)
    best_param = _best_param_from_grid(grid_real)
    pf_real = float(grid_real["profit_factor"].max())

    r = permutation_test_is(donchian, df, grid, best_param=best_param,
                            pf_real=pf_real, n_permutations=20, seed_base=0,
                            results_dir=str(tmp_path))
    assert os.path.exists(r["metricas"]["pf_perm_array_path"])
    assert os.path.exists(r["metricas"]["histograma"])
    assert r["metricas"]["histograma"].endswith(".png")
    arr = np.load(r["metricas"]["pf_perm_array_path"])
    assert arr.shape == (20,)


def test_permutation_test_is_pf_real_infinito_levanta(tmp_path):
    df = _df_tendencia(n=50)

    def fake(df, **p):
        return pd.Series(0.0, index=df.index)

    with pytest.raises(ValueError):
        permutation_test_is(fake, df, {"a": [1], "b": [1]},
                            best_param={"a": 1, "b": 1},
                            pf_real=float("inf"), n_permutations=5,
                            results_dir=str(tmp_path))
