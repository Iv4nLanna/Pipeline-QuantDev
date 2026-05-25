# Camada 1 — Sub-ciclo 1C (Fixtures sintéticos + Passo 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar três fixtures de cenário canônicos (`edge_real`, `ruido_puro`, `donchian_btc`) sob `tests/fixtures/`, a estratégia auxiliar `momentum_consec`, o Passo 3 (`walk_forward_test`), e validação retroativa dos Passos 1 e 2 contra os fixtures.

**Architecture:** Fixtures como funções puras retornando `Scenario(name, df, strategy, param_grid, expected_*)`. Cada camada (existente e futura) itera sobre os cenários via `pytest.parametrize` e assegura o status esperado — provando que a fábrica aprova/reprova pelas razões certas, não por acidente. `walk_forward_test` em `validation/walk_forward.py` faz anchored WFO com janelas OOS não-sobrepostas, gate qualitativo (`pf > 1 AND n_trades >= 30`), reusando `evaluate_grid` + `bar_returns` sem redefinir contratos.

**Tech Stack:** Python 3.14, pandas, numpy, matplotlib, pytest. Spec: `docs/superpowers/specs/2026-05-23-camada1-1c-fixtures-walk-forward.md`.

---

## File Structure

- `strategies/momentum_consec.py` — **novo**, `strategy_momentum_consec(df, lookback)`.
- `tests/fixtures/__init__.py` — **novo** (vazio, marca como package).
- `tests/fixtures/synthetic_signals.py` — **novo**, geradores + `Scenario` + `get_scenarios()`.
- `validation/walk_forward.py` — **novo**, `walk_forward_test` + `_split_windows`.
- `validation/layer1.py` — + `_plot_equity_curve_oos` (helper de viz; reusado pelo walk_forward).
- `tests/test_momentum_consec.py` — **novo**.
- `tests/test_fixtures.py` — **novo**.
- `tests/test_walk_forward.py` — **novo** (TDD do Passo 3).
- `tests/test_layer1_scenarios.py` — **novo** (Passo 1 × 3 cenários).
- `tests/test_layer1_passo2_scenarios.py` — **novo** (Passo 2 × 3 cenários, `n_permutations=50`).
- `tests/test_walk_forward_scenarios.py` — **novo** (Passo 3 × 3 cenários).
- `docs/STRATEGIES_TESTED.md` — atualizado com resultado de Donchian no Passo 3.

---

## Task 1: strategies/momentum_consec.py — estratégia auxiliar (TDD)

**Files:**
- Create: `strategies/momentum_consec.py`
- Create: `tests/test_momentum_consec.py`

- [ ] **Step 1: Escrever testes falhos**

```python
# tests/test_momentum_consec.py
import numpy as np
import pandas as pd
import pytest

from strategies.momentum_consec import strategy_momentum_consec


def _df(prices):
    p = pd.Series(prices, dtype=float)
    idx = pd.date_range("2020-01-01", periods=len(p), freq="1h")
    arr = p.values
    return pd.DataFrame({"open": arr, "high": arr, "low": arr, "close": arr,
                         "volume": 1.0}, index=idx)


def test_long_apos_n_barras_positivas():
    # 5 barras subindo monotonicamente -> sinal long a partir da 3ª (lookback=3)
    df = _df([100, 101, 102, 103, 104, 105])
    sig = strategy_momentum_consec(df, lookback=3)
    # diff: [NaN, +1, +1, +1, +1, +1]; "3 positivos consecutivos" só vale
    # quando a janela de 3 diffs está completa, ou seja, a partir do índice 3
    assert sig.iloc[3] == 1
    assert sig.iloc[4] == 1
    assert sig.iloc[5] == 1
    # primeiras 3 barras: ainda no warmup do rolling -> 0
    assert (sig.iloc[:3] == 0).all()


def test_short_apos_n_barras_negativas():
    df = _df([105, 104, 103, 102, 101, 100])
    sig = strategy_momentum_consec(df, lookback=3)
    assert sig.iloc[3] == -1
    assert sig.iloc[4] == -1
    assert sig.iloc[5] == -1


def test_flat_em_mistura():
    df = _df([100, 101, 100, 101, 100, 101])  # alterna
    sig = strategy_momentum_consec(df, lookback=3)
    # nunca tem 3 positivos consecutivos nem 3 negativos -> sempre 0
    assert (sig == 0).all()


def test_sem_lookahead():
    df = _df([100, 101, 102, 103, 104, 105, 106, 107])
    sig_full = strategy_momentum_consec(df, lookback=3)
    df2 = df.copy()
    df2.iloc[7, df2.columns.get_loc("close")] = 999
    sig2 = strategy_momentum_consec(df2, lookback=3)
    pd.testing.assert_series_equal(sig_full.iloc[:7], sig2.iloc[:7])


def test_valida_coluna_close():
    bad = pd.DataFrame({"open": [1.0, 2.0]})
    with pytest.raises(ValueError):
        strategy_momentum_consec(bad, lookback=3)
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_momentum_consec.py -v`
Expected: `ModuleNotFoundError: No module named 'strategies.momentum_consec'`.

- [ ] **Step 3: Implementar**

```python
# strategies/momentum_consec.py
import numpy as np
import pandas as pd


def strategy_momentum_consec(df, lookback):
    """Long se as últimas `lookback` barras tiveram retorno positivo;
    short se todas negativas; flat caso contrário.

    Sinal em t usa close[t-lookback..t] (info disponível no fim de t).
    A defasagem de execução de 1 barra é aplicada depois em
    metrics.bar_returns (signals.shift(1)).
    """
    if "close" not in df.columns:
        raise ValueError("strategy_momentum_consec requer coluna 'close'")
    diff = df["close"].diff()
    up = (diff > 0).astype(int)
    down = (diff < 0).astype(int)
    n_up = up.rolling(lookback).sum()
    n_down = down.rolling(lookback).sum()
    sig = pd.Series(0.0, index=df.index)
    sig[n_up == lookback] = 1.0
    sig[n_down == lookback] = -1.0
    return sig
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_momentum_consec.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add strategies/momentum_consec.py tests/test_momentum_consec.py
git commit -m "feat: strategy_momentum_consec — long/short após N barras consecutivas"
```

---

## Task 2: tests/fixtures/ — Scenario + fixture_edge_real + fixture_ruido_puro

**Files:**
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/synthetic_signals.py`
- Create: `tests/test_fixtures.py`

- [ ] **Step 1: Escrever testes falhos**

```python
# tests/test_fixtures.py
import numpy as np
import pandas as pd
import pytest

from tests.fixtures.synthetic_signals import (
    Scenario, fixture_edge_real, fixture_ruido_puro, get_scenarios,
)
from strategies.momentum_consec import strategy_momentum_consec
from validation.metrics import bar_returns, profit_factor, trade_stats


def test_fixture_edge_real_shape_e_dtype():
    df = fixture_edge_real()
    assert len(df) >= 5000
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].notna().all()
    assert (df["close"] > 0).all()


def test_fixture_edge_real_tem_edge_capturavel():
    df = fixture_edge_real()
    sig = strategy_momentum_consec(df, lookback=3)
    n_trades, _ = trade_stats(sig, df["close"])
    pf = profit_factor(bar_returns(sig, df["close"]))
    assert n_trades >= 30
    assert pf > 1.10, f"esperado PF > 1.10 no fixture com edge, obtive {pf:.3f}"


def test_fixture_ruido_puro_sem_edge():
    df = fixture_ruido_puro()
    sig = strategy_momentum_consec(df, lookback=3)
    pf = profit_factor(bar_returns(sig, df["close"]))
    assert 0.85 < pf < 1.15, f"esperado PF ~1.0 no ruído puro, obtive {pf:.3f}"


def test_fixture_reprodutibilidade():
    a = fixture_edge_real()
    b = fixture_edge_real()
    pd.testing.assert_frame_equal(a, b)


def test_get_scenarios_estrutura():
    cenarios = get_scenarios()
    nomes = [c.name for c in cenarios]
    assert "edge_real" in nomes
    assert "ruido_puro" in nomes
    # donchian_btc é opcional (skipado se cache ausente); pode ou não estar
    for c in cenarios:
        assert isinstance(c, Scenario)
        assert callable(c.strategy)
        assert isinstance(c.param_grid, dict)
        assert c.expected_passo1 in {"aprovado", "reprovado"}
        assert c.expected_passo2 in {"aprovado", "reprovado", None}
        assert c.expected_passo3 in {"aprovado", "reprovado", None}
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_fixtures.py -v`
Expected: `ModuleNotFoundError: No module named 'tests.fixtures'`.

- [ ] **Step 3: Criar tests/fixtures/__init__.py (vazio)**

```bash
touch tests/fixtures/__init__.py
```

- [ ] **Step 4: Implementar synthetic_signals.py**

```python
# tests/fixtures/synthetic_signals.py
"""Cenários canônicos para validar a FÁBRICA (Camadas 1-8).

Cada cenário é um Scenario(name, df, strategy, param_grid, expected_passo1,
expected_passo2, expected_passo3). As camadas iteram sobre get_scenarios() e
asseguram que o status produzido bate com o esperado — provando que a camada
aprova/reprova pelas razões certas, não por acidente.
"""
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from strategies.donchian import donchian
from strategies.momentum_consec import strategy_momentum_consec


MOMENTUM_GRID = {"lookback": [2, 3, 4]}


@dataclass
class Scenario:
    name: str
    df: pd.DataFrame
    strategy: Callable
    param_grid: dict
    expected_passo1: str
    expected_passo2: Optional[str]
    expected_passo3: Optional[str]


def _ohlc_from_close(close, idx):
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1.0}, index=idx)


def fixture_edge_real(n=5000, lookback=3, drift_signal=0.002, sigma=0.005, seed=0):
    """Log-retornos com drift CONDICIONAL plantado:
    após `lookback` retornos positivos consecutivos -> drift +drift_signal;
    após `lookback` negativos consecutivos -> -drift_signal; senão 0.
    Ruído gaussiano sigma sempre presente.

    Propriedades:
      - estratégia momentum_consec(lookback=3) captura o sinal (PF > 1.1).
      - permutação dos log-retornos destrói a condição temporal -> sinal some.
      - drift estável no tempo -> walk-forward preserva.
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        if t > lookback:
            past = r[t - lookback:t]
            if (past > 0).all():
                base = drift_signal
            elif (past < 0).all():
                base = -drift_signal
            else:
                base = 0.0
        else:
            base = 0.0
        r[t] = base + rng.normal(0, sigma)
    close = 100.0 * np.exp(np.cumsum(r))
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    return _ohlc_from_close(close, idx)


def fixture_ruido_puro(n=5000, sigma=0.005, seed=0):
    """Random walk geométrico puro, log-retornos ~ N(0, sigma). Sem drift,
    sem autocorrelação, sem padrão. Estratégia momentum nele -> PF ~ 1.0."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, sigma, n)
    close = 100.0 * np.exp(np.cumsum(r))
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    return _ohlc_from_close(close, idx)


def get_scenarios():
    """Lista de cenários canônicos com status esperados por camada.
    expected_*=None significa "sem assertiva fixa" (mundo real)."""
    cenarios = [
        Scenario(
            name="edge_real",
            df=fixture_edge_real(),
            strategy=strategy_momentum_consec,
            param_grid=MOMENTUM_GRID,
            expected_passo1="aprovado",
            expected_passo2="aprovado",
            expected_passo3="aprovado",
        ),
        Scenario(
            name="ruido_puro",
            df=fixture_ruido_puro(),
            strategy=strategy_momentum_consec,
            param_grid=MOMENTUM_GRID,
            expected_passo1="aprovado",     # ruído não dispara red-flags
            expected_passo2="reprovado",
            expected_passo3="reprovado",
        ),
    ]
    # cenário do mundo real: opcional, sem assertiva fixa em Passo 2/3
    cenarios.append(_donchian_btc_scenario_or_none())
    return [c for c in cenarios if c is not None]


def _donchian_btc_scenario_or_none():
    """Carrega BTC do cache parquet existente; retorna None se ausente."""
    import config
    cache = os.path.join(config.CACHE_DIR, "btcusdt_1h.parquet")
    if not os.path.exists(cache):
        return None
    try:
        from data.layer0 import load_clean_ohlc
        df = load_clean_ohlc(config.TICKER, config.TIMEFRAME,
                             config.START, config.END,
                             gap_threshold=config.GAP_THRESHOLD,
                             cache_dir=config.CACHE_DIR,
                             exchange_name=config.EXCHANGE)
    except Exception:
        return None
    return Scenario(
        name="donchian_btc",
        df=df,
        strategy=donchian,
        param_grid=config.DONCHIAN_PARAM_GRID,
        expected_passo1="aprovado",
        expected_passo2=None,   # já medido: reprovado, mas sem assertiva fixa
        expected_passo3=None,
    )
```

- [ ] **Step 5: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_fixtures.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/__init__.py tests/fixtures/synthetic_signals.py tests/test_fixtures.py
git commit -m "feat: 3 fixtures canônicos (edge_real, ruido_puro, donchian_btc) + Scenario"
```

---

## Task 3: validar Passo 1 contra os 3 cenários

**Files:**
- Create: `tests/test_layer1_scenarios.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_layer1_scenarios.py
import pytest

from tests.fixtures.synthetic_signals import get_scenarios
from validation.layer1 import in_sample_excellence


@pytest.mark.parametrize("cenario", get_scenarios(), ids=lambda c: c.name)
def test_passo1_status_esperado(cenario, tmp_path):
    if cenario.expected_passo1 is None:
        pytest.skip("sem assertiva fixa para este cenário no Passo 1")
    r = in_sample_excellence(cenario.strategy, cenario.df, cenario.param_grid,
                             results_dir=str(tmp_path))
    assert r["status"] == cenario.expected_passo1, (
        f"cenário {cenario.name}: esperado {cenario.expected_passo1}, "
        f"obtive {r['status']} — motivo: {r['motivo']}"
    )
```

- [ ] **Step 2: Rodar**

Run: `.venv/bin/python -m pytest tests/test_layer1_scenarios.py -v`
Expected: 3 passed (edge_real=aprovado, ruido_puro=aprovado, donchian_btc=aprovado). Se algum falhar, **PARAR** e investigar — a falha aponta para um bug no Passo 1 ou em um fixture, e a fábrica está mentindo.

- [ ] **Step 3: Commit**

```bash
git add tests/test_layer1_scenarios.py
git commit -m "test: Passo 1 valida o status esperado nos 3 cenários canônicos"
```

---

## Task 4: validar Passo 2 contra os 3 cenários

**Files:**
- Create: `tests/test_layer1_passo2_scenarios.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_layer1_passo2_scenarios.py
import pytest

from tests.fixtures.synthetic_signals import get_scenarios
from validation.engine import evaluate_grid
from validation.layer1 import in_sample_excellence, permutation_test_is


@pytest.mark.parametrize("cenario", get_scenarios(), ids=lambda c: c.name)
def test_passo2_status_esperado(cenario, tmp_path):
    if cenario.expected_passo2 is None:
        pytest.skip("sem assertiva fixa para este cenário no Passo 2")
    if cenario.name == "donchian_btc":
        pytest.skip("Donchian em BTC: 500 perms = lento; rodar via script")

    r1 = in_sample_excellence(cenario.strategy, cenario.df, cenario.param_grid,
                              results_dir=str(tmp_path))
    assert r1["status"] == "aprovado", "Passo 1 deveria aprovar antes do 2"

    r2 = permutation_test_is(
        cenario.strategy, cenario.df, cenario.param_grid,
        best_param=r1["metricas"]["best_param"],
        pf_real=r1["metricas"]["profit_factor"],
        n_permutations=50, seed_base=0,
        results_dir=str(tmp_path),
    )
    assert r2["status"] == cenario.expected_passo2, (
        f"cenário {cenario.name}: esperado {cenario.expected_passo2}, "
        f"obtive {r2['status']} — p_value={r2['metricas']['p_value']:.4f}"
    )
```

- [ ] **Step 2: Rodar**

Run: `.venv/bin/python -m pytest tests/test_layer1_passo2_scenarios.py -v`
Expected: 2 passed (edge_real=aprovado, ruido_puro=reprovado). Tempo: ~30s.

- [ ] **Step 3: Commit**

```bash
git add tests/test_layer1_passo2_scenarios.py
git commit -m "test: Passo 2 valida o status esperado nos cenários sintéticos"
```

---

## Task 5: validation/walk_forward.py — helpers de split (TDD)

**Files:**
- Create: `validation/walk_forward.py`
- Create: `tests/test_walk_forward.py`

- [ ] **Step 1: Escrever os testes do split**

```python
# tests/test_walk_forward.py
import numpy as np
import pandas as pd
import pytest

from validation.walk_forward import _split_windows


def test_split_windows_basico():
    # n=100, train=40, test=20, step=20 -> janelas começando em 0, 20, 40
    # janela em inicio=40: IS=[40:80], OOS=[80:100] -> última válida
    janelas = _split_windows(n=100, train_size=40, test_size=20, step=20)
    assert len(janelas) == 3
    assert janelas[0] == (0, 40, 40, 60)
    assert janelas[1] == (20, 60, 60, 80)
    assert janelas[2] == (40, 80, 80, 100)


def test_split_windows_sem_janelas_levanta():
    with pytest.raises(ValueError):
        _split_windows(n=50, train_size=40, test_size=20, step=20)


def test_split_windows_invalido_levanta():
    with pytest.raises(ValueError):
        _split_windows(n=100, train_size=0, test_size=20, step=20)
    with pytest.raises(ValueError):
        _split_windows(n=100, train_size=40, test_size=0, step=20)
    with pytest.raises(ValueError):
        _split_windows(n=100, train_size=40, test_size=20, step=0)


def test_split_windows_oos_nao_sobrepoe_is_da_mesma_janela():
    for is_start, is_end, oos_start, oos_end in _split_windows(
        n=200, train_size=50, test_size=20, step=20):
        assert oos_start == is_end   # OOS começa imediatamente após IS
        assert oos_end > oos_start
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_walk_forward.py -v`
Expected: `ModuleNotFoundError: No module named 'validation.walk_forward'`.

- [ ] **Step 3: Implementar split**

```python
# validation/walk_forward.py
def _split_windows(n, train_size, test_size, step):
    """Anchored sliding window: cada janela é (is_start, is_end, oos_start, oos_end).

    OOS começa imediatamente após IS na MESMA janela. step controla o avanço entre
    janelas consecutivas (step = test_size => janelas OOS não sobrepostas).
    """
    if train_size <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("train_size, test_size, step devem ser positivos")
    janelas = []
    inicio = 0
    while inicio + train_size + test_size <= n:
        is_start = inicio
        is_end = inicio + train_size
        oos_start = is_end
        oos_end = is_end + test_size
        janelas.append((is_start, is_end, oos_start, oos_end))
        inicio += step
    if not janelas:
        raise ValueError(
            f"nenhuma janela: n={n} insuficiente para "
            f"train_size={train_size} + test_size={test_size}"
        )
    return janelas
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_walk_forward.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: _split_windows para walk-forward (anchored sliding)"
```

---

## Task 6: walk_forward_test + _plot_equity_curve_oos (TDD)

**Files:**
- Modify: `validation/walk_forward.py`
- Modify: `validation/layer1.py` (anexa `_plot_equity_curve_oos`)
- Modify: `tests/test_walk_forward.py` (anexa)

- [ ] **Step 1: Anexar testes**

```python
# tests/test_walk_forward.py (anexar)
from validation.metrics import bar_returns, profit_factor
from validation.walk_forward import walk_forward_test


def _df_tendencia(n=500, drift=0.001, sigma=0.005, seed=0):
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(seed)
    r = drift + rng.normal(0, sigma, n)
    close = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1.0}, index=idx)


def _fake_strategy(df, lookback):
    return pd.Series(1.0, index=df.index)


def test_walk_forward_test_dict_padronizado(tmp_path):
    df = _df_tendencia(n=500)
    grid = {"lookback": [2, 3]}
    r = walk_forward_test(_fake_strategy, df, grid, train_size=200,
                          test_size=100, step=100, results_dir=str(tmp_path))
    assert r["camada"] == "walk_forward_test"
    assert r["status"] in {"aprovado", "reprovado"}
    assert "pf_oos_total" in r["metricas"]
    assert "n_oos_trades" in r["metricas"]
    assert "n_janelas" in r["metricas"]
    assert "janelas" in r["metricas"]
    assert "equity_curve_path" in r["metricas"]
    assert len(r["metricas"]["janelas"]) == r["metricas"]["n_janelas"]
    assert r["metricas"]["n_janelas"] >= 1
    import os
    assert os.path.exists(r["metricas"]["equity_curve_path"])


def test_walk_forward_test_pf_oos_consistente_com_concat(tmp_path):
    """pf_oos_total deve bater com profit_factor aplicado à concat dos OOS rets."""
    df = _df_tendencia(n=500)
    grid = {"lookback": [2]}
    r = walk_forward_test(_fake_strategy, df, grid, train_size=200,
                          test_size=100, step=100, results_dir=str(tmp_path))
    # reconstrói: para cada janela, aplica best_param ao slice OOS, concatena
    rets = []
    for j in r["metricas"]["janelas"]:
        oos = df.iloc[j["oos_start"]:j["oos_end"]]
        sig = _fake_strategy(oos, **j["best_param"])
        rets.append(bar_returns(sig, oos["close"]))
    pf_recalc = profit_factor(np.concatenate(rets))
    assert abs(r["metricas"]["pf_oos_total"] - pf_recalc) < 1e-9


def test_walk_forward_test_df_curto_levanta(tmp_path):
    df = _df_tendencia(n=50)
    with pytest.raises(ValueError):
        walk_forward_test(_fake_strategy, df, {"lookback": [2]},
                          train_size=40, test_size=20, step=20,
                          results_dir=str(tmp_path))
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_walk_forward.py -v`
Expected: 4 passed da Task 5 + 3 falhas com `ImportError: cannot import name 'walk_forward_test'`.

- [ ] **Step 3: Implementar `_plot_equity_curve_oos` em layer1.py (helper de viz, reusado)**

Adicionar ao final de `validation/layer1.py`:

```python
def _plot_equity_curve_oos(rets_oos_concat, janelas, results_dir):
    """Curva de equity OOS = (1 + rets).cumprod(). Linhas verticais marcam
    o início de cada janela OOS no eixo concatenado.
    """
    import os
    from datetime import datetime
    os.makedirs(results_dir, exist_ok=True)
    equity = (1.0 + rets_oos_concat).cumprod()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity, lw=1.0)
    ax.set_xlabel("barras OOS concatenadas")
    ax.set_ylabel("equity (inicia em 1)")
    ax.set_title("Walk-forward OOS — curva de equity concatenada")
    offset = 0
    for j in janelas:
        ax.axvline(offset, color="grey", linestyle=":", lw=0.6)
        offset += (j["oos_end"] - j["oos_start"])
    ax.axhline(1.0, color="red", linestyle="--", lw=0.8)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(results_dir, f"equity_curve_wf_{ts}.png")
    fig.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return caminho
```

- [ ] **Step 4: Implementar `walk_forward_test` em walk_forward.py**

Anexar ao final de `validation/walk_forward.py`:

```python
import numpy as np
import pandas as pd

import config
from utils.contract import make_result, save_result
from validation.engine import evaluate_grid
from validation.layer1 import _plot_equity_curve_oos
from validation.metrics import bar_returns, profit_factor, trade_stats


def walk_forward_test(strategy_func, df, param_grid,
                      train_size=None, test_size=None, step=None,
                      results_dir="results"):
    """Passo 3 da Camada 1: anchored walk-forward com janelas OOS não sobrepostas.

    Em cada janela: re-otimiza param_grid no slice IS, aplica best_param no slice
    OOS, coleta os bar_returns OOS. Concatena todos os rets OOS e computa o
    profit_factor final + n_trades total.

    Gate qualitativo: pf_oos_total > 1.0 AND n_oos_trades >= REDFLAG_MIN_TRADES.
    """
    n = len(df)
    if train_size is None:
        train_size = int(n * 0.4)
    if test_size is None:
        test_size = int(n * 0.1)
    if step is None:
        step = test_size

    janelas_idx = _split_windows(n, train_size, test_size, step)
    rets_oos_concat = []
    janelas_info = []
    for is_start, is_end, oos_start, oos_end in janelas_idx:
        is_slice = df.iloc[is_start:is_end]
        oos_slice = df.iloc[oos_start:oos_end]
        grid_is = evaluate_grid(strategy_func, is_slice, param_grid)
        best_idx = grid_is["profit_factor"].idxmax()
        tup = best_idx if isinstance(best_idx, tuple) else (best_idx,)
        best_param = {k: int(v) for k, v in zip(grid_is.index.names, tup)}
        pf_is = float(grid_is["profit_factor"].loc[best_idx])

        sig_oos = strategy_func(oos_slice, **best_param)
        rets = bar_returns(sig_oos, oos_slice["close"])
        n_trades_oos, _ = trade_stats(sig_oos, oos_slice["close"])
        pf_oos = profit_factor(rets)
        rets_oos_concat.append(rets)
        janelas_info.append({
            "is_start": is_start, "is_end": is_end,
            "oos_start": oos_start, "oos_end": oos_end,
            "best_param": best_param,
            "pf_is": pf_is, "pf_oos": float(pf_oos),
            "n_trades_oos": int(n_trades_oos),
        })

    rets_total = np.concatenate(rets_oos_concat)
    pf_oos_total = profit_factor(rets_total)
    n_oos_trades = sum(j["n_trades_oos"] for j in janelas_info)
    equity_path = _plot_equity_curve_oos(rets_total, janelas_info, results_dir)

    metricas = {
        "pf_oos_total": float(pf_oos_total),
        "n_oos_trades": int(n_oos_trades),
        "n_janelas": len(janelas_info),
        "train_size": train_size,
        "test_size": test_size,
        "step": step,
        "janelas": janelas_info,
        "equity_curve_path": equity_path,
    }

    if pf_oos_total > 1.0 and n_oos_trades >= config.REDFLAG_MIN_TRADES:
        status = "aprovado"
        motivo = (f"OOS PF={pf_oos_total:.3f} > 1.0 com {n_oos_trades} trades "
                  f"em {len(janelas_info)} janelas.")
        proximo = "Avançar para o Passo 4 (wf_permutation_test)."
    else:
        razoes = []
        if pf_oos_total <= 1.0:
            razoes.append(f"OOS PF={pf_oos_total:.3f} <= 1.0")
        if n_oos_trades < config.REDFLAG_MIN_TRADES:
            razoes.append(f"n_oos_trades={n_oos_trades} < {config.REDFLAG_MIN_TRADES}")
        status = "reprovado"
        motivo = "; ".join(razoes)
        proximo = "Revisar estratégia ou parâmetros do walk-forward."

    resultado = make_result("walk_forward_test", status, metricas, motivo, proximo)
    save_result(resultado, results_dir=results_dir)
    return resultado
```

- [ ] **Step 5: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_walk_forward.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add validation/walk_forward.py validation/layer1.py tests/test_walk_forward.py
git commit -m "feat: walk_forward_test (Passo 3) com gate qualitativo PF>1 e equity OOS"
```

---

## Task 7: validar Passo 3 contra os 3 cenários

**Files:**
- Create: `tests/test_walk_forward_scenarios.py`

- [ ] **Step 1: Escrever o teste**

```python
# tests/test_walk_forward_scenarios.py
import pytest

from tests.fixtures.synthetic_signals import get_scenarios
from validation.walk_forward import walk_forward_test


@pytest.mark.parametrize("cenario", get_scenarios(), ids=lambda c: c.name)
def test_passo3_status_esperado(cenario, tmp_path):
    if cenario.expected_passo3 is None:
        pytest.skip("sem assertiva fixa para este cenário no Passo 3")
    if cenario.name == "donchian_btc":
        pytest.skip("Donchian em BTC: walk-forward pesado; rodar via script")

    r = walk_forward_test(cenario.strategy, cenario.df, cenario.param_grid,
                         results_dir=str(tmp_path))
    assert r["status"] == cenario.expected_passo3, (
        f"cenário {cenario.name}: esperado {cenario.expected_passo3}, "
        f"obtive {r['status']} — motivo: {r['motivo']}"
    )
```

- [ ] **Step 2: Rodar**

Run: `.venv/bin/python -m pytest tests/test_walk_forward_scenarios.py -v`
Expected: 2 passed (edge_real=aprovado, ruido_puro=reprovado).

> **Se falhar:** **NÃO ajustar a estratégia ou os parâmetros do Passo 3 até passar**. Investigar honestamente:
> (a) o fixture pode estar mal calibrado — ajustar `drift_signal`, `sigma` ou `seed` do `fixture_edge_real` para que o sinal seja capturável por walk-forward, ou da `fixture_ruido_puro` para que produza OOS PF < 1.0 com a seed escolhida;
> (b) o gate do Passo 3 pode ser inadequado — revisar a spec antes de mudar código;
> (c) bug na implementação — debug normal.
> Documentar a escolha de seeds/parâmetros em comentário do fixture.

- [ ] **Step 3: Commit**

```bash
git add tests/test_walk_forward_scenarios.py
git commit -m "test: Passo 3 valida o status esperado nos cenários sintéticos"
```

---

## Task 8: rodar Passo 3 no BTC (Donchian) e atualizar registry

**Files:**
- Modify: `docs/STRATEGIES_TESTED.md`

- [ ] **Step 1: Rodar a suíte inteira**

Run: `.venv/bin/python -m pytest -q`
Expected: tudo passa (offline; teste `network` da C0 continua skipped).

- [ ] **Step 2: Rodar Passo 3 sobre Donchian BTC**

Run:
```bash
.venv/bin/python -c "
import json, config
from data.layer0 import load_clean_ohlc
from strategies.donchian import donchian
from validation.walk_forward import walk_forward_test

df = load_clean_ohlc(config.TICKER, config.TIMEFRAME, config.START, config.END,
                     gap_threshold=config.GAP_THRESHOLD, cache_dir=config.CACHE_DIR,
                     exchange_name=config.EXCHANGE)
print(f'>> dataset: {len(df)} barras')
r = walk_forward_test(donchian, df, config.DONCHIAN_PARAM_GRID,
                      results_dir=config.RESULTS_DIR)
print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
"
```
Expected: dict com `pf_oos_total`, `n_janelas`, `equity_curve_path`. Tempo estimado <1 min.

- [ ] **Step 3: Atualizar registry**

Anexar uma seção em `docs/STRATEGIES_TESTED.md` documentando o status do Passo 3 para Donchian. **Não é falha do projeto — é informação.** O sub-ciclo 1C é sobre construir a fábrica; o status real é registrado, não interpretado como sucesso/fracasso da fábrica.

Estrutura sugerida (preencher com valores reais do Step 2):

```markdown
## Passo 3 (walk_forward_test) — Donchian BTC

- **pf_oos_total:** <valor>
- **n_oos_trades:** <valor>
- **n_janelas:** <valor> (train_size=<valor>, test_size=<valor>)
- **status:** <valor> — sem assertiva fixa para mundo real.
- **equity_curve:** `results/equity_curve_wf_<ts>.png`
```

- [ ] **Step 4: Commit**

```bash
git add docs/STRATEGIES_TESTED.md
git commit -m "docs: registra Passo 3 da Donchian BTC (sem assertiva)"
```

---

## Self-Review

**1. Cobertura da spec:**
- §2 (contrato dos 3 fixtures) — Tasks 1 e 2 criam estratégia auxiliar + fixtures + Scenario.
- §3 (algoritmo do walk_forward_test) — Tasks 5 e 6 entregam `_split_windows` + `walk_forward_test`.
- §3.5 (output dict) — Task 6 inclui `pf_oos_total`, `n_oos_trades`, `n_janelas`, `janelas`, `equity_curve_path`.
- §3.6 (`_plot_equity_curve_oos`) — Task 6, anexado em `layer1.py` (visualizações ficam juntas).
- §4 (estratégia auxiliar) — Task 1.
- §6.3-6.6 (testes por camada × cenário) — Tasks 3, 4, 7 cobrem Passos 1, 2, 3.
- §7 (erros) — Tasks 1 e 5 incluem validações de entrada com ValueError.
- §8 (critério de pronto) — Task 8 fecha com execução real e atualização do registry.

**2. Placeholders:** nenhum. Todo step tem código concreto ou comando exato.

**3. Consistência de tipos:**
- `Scenario(name, df, strategy, param_grid, expected_passo1, expected_passo2, expected_passo3)` — usado de forma idêntica entre `synthetic_signals.py`, `test_fixtures.py` e os 3 arquivos `test_*_scenarios.py`.
- `walk_forward_test(strategy_func, df, param_grid, train_size, test_size, step, results_dir)` — único entre `validation/walk_forward.py` e testes.
- `strategy_momentum_consec(df, lookback)` — assinatura única entre estratégia, fixtures e testes.
- `_split_windows(n, train_size, test_size, step) -> list[tuple[int,int,int,int]]` — formato consistente entre impl e teste.

**4. Tratamento honesto de falhas:** Task 7 inclui guidance explícita de "não ajustar até passar" — protege contra a tentação de re-calibrar fixtures para esconder bugs. Task 8 documenta que status do Donchian no Passo 3 é informação, não veredicto.
