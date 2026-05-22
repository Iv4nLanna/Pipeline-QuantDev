# Camada 1 — Sub-ciclo 1A (Fundação) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar a fundação da Camada 1 — estratégia Donchian, motor de métricas barra-a-barra, varredura de param_grid e o Passo 1 (`in_sample_excellence`) com heatmap de parâmetros.

**Architecture:** Funções puras sobre `pd.DataFrame`/`pd.Series`, contrato padronizado (`utils/contract.py`) nas funções-portão. VectorBT adotado como dependência estrutural (smoke-test prova o numba/JIT em Python 3.14); o `profit_factor` é sempre calculado a partir de retornos barra-a-barra, com a convenção anti-lookahead `posição = sinal.shift(1)`.

**Tech Stack:** Python 3.14, venv, pandas, numpy, matplotlib, vectorbt, numba, pytest. Spec: `docs/superpowers/specs/2026-05-22-camada1-1a-fundacao-design.md`.

---

## File Structure

- `requirements.txt` — + `vectorbt`, `numba`.
- `config.py` — + `DONCHIAN_PARAM_GRID`, `PF_MIN`, red-flags.
- `validation/metrics.py` — `bar_returns`, `profit_factor`, `trade_stats` (núcleo reusado pelo 1B).
- `strategies/donchian.py` — `donchian` (interface de estratégia `-> Series[-1,0,1]`).
- `validation/engine.py` — `evaluate_grid` (varredura do param_grid).
- `validation/layer1.py` — `in_sample_excellence` + `_plot_param_heatmap`.
- `data/layer0.py` — + `load_clean_ohlc` (reusa `fetch_data` + `clean_data`).
- `tests/test_metrics.py`, `test_donchian.py`, `test_engine.py`, `test_layer1.py`, `test_vbt_smoke.py`.

---

## Task 1: Dependências (vectorbt, numba) e smoke-test do JIT

**Files:**
- Modify: `requirements.txt`
- Test: `tests/test_vbt_smoke.py`

- [ ] **Step 1: Adicionar dependências ao requirements.txt**

Anexar ao final de `requirements.txt`:
```
vectorbt>=1.0
numba>=0.65
```

- [ ] **Step 2: Instalar**

Run:
```bash
cd /home/ivan/trading_pipeline
.venv/bin/pip install -q -r requirements.txt
```
Expected: instala sem erro (puxa `vectorbt`, `numba`, `llvmlite` e dependências transitivas).

- [ ] **Step 3: Escrever o smoke-test**

```python
# tests/test_vbt_smoke.py
def test_numba_jit_executa():
    from numba import njit

    @njit
    def soma(a, b):
        return a + b

    assert soma(2, 3) == 5


def test_vectorbt_importa():
    import vectorbt as vbt

    assert hasattr(vbt, "Portfolio")
```

- [ ] **Step 4: Rodar o smoke-test**

Run: `.venv/bin/python -m pytest tests/test_vbt_smoke.py -v`
Expected: 2 passed. (Se `test_numba_jit_executa` falhar, o numba não compila em Python 3.14 — PARAR e reportar; toda a escolha de engine depende disso.)

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/test_vbt_smoke.py
git commit -m "chore: adiciona vectorbt+numba e smoke-test do JIT em Python 3.14"
```

---

## Task 2: config.py — param_grid e thresholds

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Anexar ao config.py**

```python
# Camada 1 — validação
DONCHIAN_PARAM_GRID = {
    "entry_lookback": [20, 30, 40, 60, 80, 100],
    "exit_lookback": [10, 20, 30, 40],
}
PF_MIN = 1.05                # profit factor mínimo para aprovar in-sample
REDFLAG_WIN_RATE = 0.95      # win_rate acima disto = overfit suspeito
REDFLAG_MIN_TRADES = 30      # menos trades que isto = sem poder estatístico
REDFLAG_MAX_PF = 5.0         # PF acima disto = implausível em dados reais
```

- [ ] **Step 2: Verificar import**

Run: `.venv/bin/python -c "import config; print(config.PF_MIN, config.DONCHIAN_PARAM_GRID['entry_lookback'])"`
Expected: `1.05 [20, 30, 40, 60, 80, 100]`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: config da Camada 1 (param_grid Donchian + thresholds)"
```

---

## Task 3: metrics.bar_returns

**Files:**
- Create: `validation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Escrever o teste falho**

```python
# tests/test_metrics.py
import numpy as np
import pandas as pd
from validation.metrics import bar_returns


def test_bar_returns_long_e_short_com_shift():
    close = pd.Series([100.0, 110.0, 99.0])
    signals = pd.Series([1, 1, -1])
    # pos = signals.shift(1) = [0, 1, 1] -> barra0 flat; barra1 long; barra2 long
    r = bar_returns(signals, close)
    assert r[0] == 0.0
    assert abs(r[1] - 0.10) < 1e-9          # long: 110/100 - 1
    assert abs(r[2] - (99 / 110 - 1)) < 1e-9  # long: 99/110 - 1


def test_bar_returns_short():
    close = pd.Series([100.0, 90.0])
    signals = pd.Series([-1, -1])
    # pos = [0, -1] -> barra1 short: 100/90 - 1
    r = bar_returns(signals, close)
    assert r[0] == 0.0
    assert abs(r[1] - (100 / 90 - 1)) < 1e-9
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -k bar_returns -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'validation.metrics'`.

- [ ] **Step 3: Implementar bar_returns**

```python
# validation/metrics.py
import numpy as np
import pandas as pd


def bar_returns(signals, close):
    """Retornos barra-a-barra com defasagem de execução de 1 barra (anti-lookahead).

    posição efetiva na barra t = signals[t-1]. long: close[t]/close[t-1]-1;
    short: close[t-1]/close[t]-1; flat: 0.
    """
    signals = pd.Series(signals).reset_index(drop=True).astype(float)
    close = pd.Series(close).reset_index(drop=True).astype(float)
    pos = signals.shift(1).fillna(0.0)
    prev = close.shift(1)
    long_ret = close / prev - 1.0
    short_ret = prev / close - 1.0
    ret = pd.Series(0.0, index=close.index)
    ret[pos == 1] = long_ret[pos == 1]
    ret[pos == -1] = short_ret[pos == -1]
    return ret.fillna(0.0).to_numpy()
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -k bar_returns -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/metrics.py tests/test_metrics.py
git commit -m "feat: bar_returns (retorno barra-a-barra long/short com shift anti-lookahead)"
```

---

## Task 4: metrics.profit_factor

**Files:**
- Modify: `validation/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Anexar o teste falho**

```python
# tests/test_metrics.py
from validation.metrics import profit_factor


def test_profit_factor_valor_exato():
    r = np.array([0.1, -0.05, 0.2, -0.05])
    # pos = 0.3 ; neg = -0.1 ; PF = 3.0
    assert abs(profit_factor(r) - 3.0) < 1e-9


def test_profit_factor_sem_perdas_e_sem_ganhos():
    assert profit_factor(np.array([0.1, 0.2])) == float("inf")
    assert profit_factor(np.array([-0.1, -0.2])) == 0.0
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -k profit_factor -v`
Expected: FAIL com `ImportError: cannot import name 'profit_factor'`.

- [ ] **Step 3: Implementar profit_factor (anexar em validation/metrics.py)**

```python
def profit_factor(returns):
    returns = np.asarray(returns, dtype=float)
    pos = returns[returns > 0].sum()
    neg = returns[returns < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -k profit_factor -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/metrics.py tests/test_metrics.py
git commit -m "feat: profit_factor (Σpos/|Σneg| com sentinela inf)"
```

---

## Task 5: metrics.trade_stats

**Files:**
- Modify: `validation/metrics.py`
- Test: `tests/test_metrics.py`

> **Nota de refinamento da spec §6.2:** `win_rate` exige preços para calcular o P&L do trade, então a assinatura é `trade_stats(signals, close)` (a spec listou `trade_stats(signals)`). `n_trades` conta corridas máximas de posição efetiva (`signals.shift(1)`) não-nula; isso é idêntico ao nº de entradas em `signals`.

- [ ] **Step 1: Anexar o teste falho**

```python
# tests/test_metrics.py
from validation.metrics import trade_stats


def test_trade_stats_conta_trades_e_win_rate():
    signals = pd.Series([0, 1, 1, 0, -1, -1, 0])
    close = pd.Series([100.0, 100.0, 110.0, 121.0, 121.0, 121.0, 100.0])
    # posição efetiva (shift1): [0,0,1,1,0,-1,-1]
    # trade1 (long, barras 2-3): (1.10*1.10)-1 = +0.21 -> win
    # trade2 (short, barras 5-6): (1.0*1.21)-1 = +0.21 -> win
    n_trades, win_rate = trade_stats(signals, close)
    assert n_trades == 2
    assert abs(win_rate - 1.0) < 1e-9


def test_trade_stats_sem_trades():
    signals = pd.Series([0, 0, 0])
    close = pd.Series([100.0, 101.0, 102.0])
    assert trade_stats(signals, close) == (0, 0.0)
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -k trade_stats -v`
Expected: FAIL com `ImportError: cannot import name 'trade_stats'`.

- [ ] **Step 3: Implementar trade_stats (anexar em validation/metrics.py)**

```python
def trade_stats(signals, close):
    signals = pd.Series(signals).reset_index(drop=True).astype(float)
    rets = pd.Series(bar_returns(signals, close))
    eff = signals.shift(1).fillna(0.0)            # posição efetiva por barra
    run_id = (eff != eff.shift(1)).cumsum()       # id de corrida (valor constante)
    out = pd.DataFrame({"eff": eff, "ret": rets, "run": run_id})
    trades = out[out["eff"] != 0].groupby("run")
    n_trades = trades.ngroups
    if n_trades == 0:
        return 0, 0.0
    pnl = trades["ret"].apply(lambda r: (1.0 + r).prod() - 1.0)
    win_rate = float((pnl > 0).mean())
    return int(n_trades), win_rate
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: todos os testes de metrics passam (6 no total).

- [ ] **Step 5: Commit**

```bash
git add validation/metrics.py tests/test_metrics.py
git commit -m "feat: trade_stats (n_trades por corrida de posição + win_rate por P&L)"
```

---

## Task 6: strategies.donchian

**Files:**
- Create: `strategies/donchian.py`
- Test: `tests/test_donchian.py`

- [ ] **Step 1: Escrever o teste falho**

```python
# tests/test_donchian.py
import numpy as np
import pandas as pd
from strategies.donchian import donchian


def _df(prices):
    p = pd.Series(prices, dtype=float)
    idx = pd.date_range("2020-01-01", periods=len(p), freq="1h")
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=idx)


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
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_donchian.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'strategies.donchian'`.

- [ ] **Step 3: Implementar donchian**

```python
# strategies/donchian.py
import numpy as np
import pandas as pd


def donchian(df, entry_lookback, exit_lookback):
    """Donchian breakout long+short (estilo Turtle) -> Series de {-1, 0, 1}.

    Canais usam apenas barras anteriores (shift(1)); a defasagem de execução
    é aplicada depois, em metrics.bar_returns.
    """
    for col in ("high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"donchian requer a coluna '{col}'")
    if df.empty:
        raise ValueError("donchian requer DataFrame não-vazio")

    entry_hi = df["high"].rolling(entry_lookback).max().shift(1).to_numpy()
    entry_lo = df["low"].rolling(entry_lookback).min().shift(1).to_numpy()
    exit_hi = df["high"].rolling(exit_lookback).max().shift(1).to_numpy()
    exit_lo = df["low"].rolling(exit_lookback).min().shift(1).to_numpy()
    c = df["close"].to_numpy()

    n = len(df)
    sig = np.zeros(n)
    pos = 0
    for t in range(n):
        if np.isnan(entry_hi[t]):
            sig[t] = 0
            continue
        if pos == 0:
            if c[t] > entry_hi[t]:
                pos = 1
            elif c[t] < entry_lo[t]:
                pos = -1
        elif pos == 1:
            if c[t] < entry_lo[t]:        # rompe canal de entrada oposto -> inverte short
                pos = -1
            elif c[t] < exit_lo[t]:        # rompe canal de saída -> flat
                pos = 0
        elif pos == -1:
            if c[t] > entry_hi[t]:         # inverte long
                pos = 1
            elif c[t] > exit_hi[t]:        # saída -> flat
                pos = 0
        sig[t] = pos
    return pd.Series(sig, index=df.index, dtype=float)
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_donchian.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add strategies/donchian.py tests/test_donchian.py
git commit -m "feat: estratégia Donchian long+short (2 canais, sem lookahead)"
```

---

## Task 7: engine.evaluate_grid

**Files:**
- Create: `validation/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Escrever o teste falho**

```python
# tests/test_engine.py
import numpy as np
import pandas as pd
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
    assert set(["profit_factor", "win_rate", "n_trades", "n_bars"]).issubset(out.columns)
    assert (out["n_bars"] == len(df)).all()


def test_evaluate_grid_param_grid_vazio():
    import pytest
    with pytest.raises(ValueError):
        evaluate_grid(donchian, _trend_df(), {})
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_engine.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'validation.engine'`.

- [ ] **Step 3: Implementar evaluate_grid**

```python
# validation/engine.py
import itertools

import pandas as pd

from validation.metrics import bar_returns, profit_factor, trade_stats


def evaluate_grid(strategy_func, df, param_grid):
    """Avalia strategy_func sobre o produto cartesiano de param_grid.

    Retorna um DataFrame indexado pelas combinações de parâmetros, com colunas
    profit_factor, win_rate, n_trades, n_bars. PF sempre via bar_returns
    (não pelo Portfolio do VectorBT), conforme a spec.
    """
    if not param_grid:
        raise ValueError("param_grid vazio")
    keys = list(param_grid.keys())
    rows = []
    for combo in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        sig = strategy_func(df, **params)
        rets = bar_returns(sig, df["close"])
        n_trades, win_rate = trade_stats(sig, df["close"])
        rows.append({
            **params,
            "profit_factor": profit_factor(rets),
            "win_rate": win_rate,
            "n_trades": n_trades,
            "n_bars": len(df),
        })
    return pd.DataFrame(rows).set_index(keys)
```

> **Nota sobre VectorBT:** o grid in-sample do 1A é pequeno (24 combinações), então a varredura é um laço direto e correto. A aceleração vetorizada do VBT é reservada para os laços de permutação do 1B (1000× re-otimização), onde "couber" de fato — conforme a spec §6.3 ("onde couber"). O VBT já está instalado e validado (Task 1).

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_engine.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/engine.py tests/test_engine.py
git commit -m "feat: evaluate_grid varre param_grid e calcula PF barra-a-barra"
```

---

## Task 8: layer1.in_sample_excellence + heatmap

**Files:**
- Create: `validation/layer1.py`
- Test: `tests/test_layer1.py`

- [ ] **Step 1: Escrever o teste falho**

```python
# tests/test_layer1.py
import os
import pandas as pd
import config
from validation.layer1 import in_sample_excellence


def _df(prices):
    p = pd.Series(prices, dtype=float)
    idx = pd.date_range("2020-01-01", periods=len(p), freq="1h")
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=idx)


def test_in_sample_excellence_aprovado(tmp_path, monkeypatch):
    # estratégia fake: sempre long, ignora params (grid de 2 chaves p/ heatmap 2D)
    def fake(df, a, b):
        return pd.Series(1.0, index=df.index)

    # afrouxa as red-flags para isolar a LÓGICA do portão (não exigir 30 trades)
    monkeypatch.setattr(config, "REDFLAG_MAX_PF", 1000.0)
    monkeypatch.setattr(config, "REDFLAG_MIN_TRADES", 1)
    monkeypatch.setattr(config, "REDFLAG_WIN_RATE", 1.01)

    df = _df([100, 101, 102, 101.5, 103, 104])  # sobe com um recuo -> PF finito > 1.05
    grid = {"a": [1, 2], "b": [3, 4]}
    r = in_sample_excellence(fake, df, grid, results_dir=str(tmp_path))

    assert r["camada"] == "in_sample_excellence"
    assert r["status"] == "aprovado"
    assert {"best_param", "profit_factor", "win_rate", "n_trades", "n_bars", "heatmap"} <= set(r["metricas"])
    assert os.path.exists(r["metricas"]["heatmap"])


def test_in_sample_excellence_reprovado_pf_baixo(tmp_path):
    def fake(df, a, b):
        return pd.Series(1.0, index=df.index)

    df = _df([100, 99, 98, 97])  # cai -> PF < 1.05
    grid = {"a": [1, 2], "b": [3, 4]}
    r = in_sample_excellence(fake, df, grid, results_dir=str(tmp_path))
    assert r["status"] == "reprovado"
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_layer1.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'validation.layer1'`.

- [ ] **Step 3: Implementar layer1**

```python
# validation/layer1.py
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from utils.contract import make_result, save_result
from validation.engine import evaluate_grid


def _plot_param_heatmap(grid_df, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    pf = grid_df["profit_factor"].replace([np.inf, -np.inf], np.nan)
    pivot = pf.unstack()  # index = 1º parâmetro, columns = 2º parâmetro
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(pivot.columns.name)
    ax.set_ylabel(pivot.index.name)
    ax.set_title("Profit factor in-sample por parâmetro")
    fig.colorbar(im, ax=ax, label="profit_factor")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(results_dir, f"param_heatmap_{ts}.png")
    fig.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return caminho


def in_sample_excellence(strategy_func, df, param_grid, results_dir="results"):
    grid_df = evaluate_grid(strategy_func, df, param_grid)
    best_idx = grid_df["profit_factor"].idxmax()
    best_row = grid_df.loc[best_idx]
    idx_vals = best_idx if isinstance(best_idx, tuple) else (best_idx,)
    best_param = {k: int(v) for k, v in zip(grid_df.index.names, idx_vals)}

    pf = float(best_row["profit_factor"])
    win_rate = float(best_row["win_rate"])
    n_trades = int(best_row["n_trades"])
    n_bars = int(best_row["n_bars"])
    heatmap = _plot_param_heatmap(grid_df, results_dir)

    metricas = {
        "best_param": best_param,
        "profit_factor": pf,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "n_bars": n_bars,
        "heatmap": heatmap,
    }

    red_flags = []
    if win_rate > config.REDFLAG_WIN_RATE:
        red_flags.append(f"win_rate {win_rate:.2%} > {config.REDFLAG_WIN_RATE:.0%}")
    if n_trades < config.REDFLAG_MIN_TRADES:
        red_flags.append(f"n_trades {n_trades} < {config.REDFLAG_MIN_TRADES}")
    if pf > config.REDFLAG_MAX_PF:
        red_flags.append(f"profit_factor {pf} > {config.REDFLAG_MAX_PF}")

    if pf > config.PF_MIN and not red_flags:
        status = "aprovado"
        motivo = f"PF={pf:.3f} com {n_trades} trades; sem red-flags."
        proximo = "Avançar para o Passo 2 (permutation_test_is)."
    else:
        status = "reprovado"
        motivo = (f"PF={pf:.3f} <= {config.PF_MIN}." if pf <= config.PF_MIN
                  else "Red-flags de overfit: " + "; ".join(red_flags))
        proximo = "Revisar estratégia/param_grid; não avançar."

    resultado = make_result("in_sample_excellence", status, metricas, motivo, proximo)
    save_result(resultado, results_dir=results_dir)
    return resultado
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_layer1.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/layer1.py tests/test_layer1.py
git commit -m "feat: in_sample_excellence (Passo 1) com gate de PF/red-flags e heatmap"
```

---

## Task 9: data.load_clean_ohlc (fluxo C0→C1)

**Files:**
- Modify: `data/layer0.py`
- Test: `tests/test_layer0.py`

- [ ] **Step 1: Anexar o teste falho**

```python
# tests/test_layer0.py
def test_load_clean_ohlc_reusa_fetch_e_clean(tmp_path, monkeypatch):
    idx = pd.date_range("2020-01-01", periods=5, freq="1h", tz="UTC")
    # inclui uma duplicata e um volume zero para provar que clean_data rodou
    raw = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 1.0, 1.0, 1.0, 1.0],
         "volume": [1.0, 1.0, 0.0, 1.0, 1.0]},
        index=idx,
    )
    monkeypatch.setattr(layer0, "fetch_data", lambda *a, **k: raw)
    out = layer0.load_clean_ohlc("BTC/USDT", "1h", "2020-01-01", "2020-01-02",
                                 cache_dir=str(tmp_path))
    assert (out["volume"] > 0).all()        # candle de volume zero removido
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_layer0.py -k load_clean_ohlc -v`
Expected: FAIL com `AttributeError: module 'data.layer0' has no attribute 'load_clean_ohlc'`.

- [ ] **Step 3: Implementar load_clean_ohlc (anexar em data/layer0.py)**

```python
def load_clean_ohlc(ticker, timeframe, start, end, gap_threshold=0.5,
                    use_cache=True, cache_dir="data/cache", exchange_name="binance"):
    """Reusa fetch_data + clean_data e devolve o DataFrame OHLCV limpo para a Camada 1."""
    bruto = fetch_data(ticker, timeframe, start, end, use_cache=use_cache,
                       cache_dir=cache_dir, exchange_name=exchange_name)
    return clean_data(bruto, gap_threshold=gap_threshold)
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_layer0.py -k load_clean_ohlc -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add data/layer0.py tests/test_layer0.py
git commit -m "feat: load_clean_ohlc expõe OHLCV limpo da C0 para a C1"
```

---

## Task 10: Execução real do Passo 1 com BTC

**Files:**
- nenhum arquivo de código novo (execução + apresentação)

- [ ] **Step 1: Rodar a suíte inteira**

Run: `.venv/bin/python -m pytest -q`
Expected: tudo passa (offline; o teste `network` da C0 continua skipped).

- [ ] **Step 2: Rodar o Passo 1 com dados reais do BTC**

Run:
```bash
.venv/bin/python -c "
import config
from data.layer0 import load_clean_ohlc
from strategies.donchian import donchian
from validation.layer1 import in_sample_excellence
import json

df = load_clean_ohlc(config.TICKER, config.TIMEFRAME, config.START, config.END,
                     gap_threshold=config.GAP_THRESHOLD, cache_dir=config.CACHE_DIR,
                     exchange_name=config.EXCHANGE)
r = in_sample_excellence(donchian, df, config.DONCHIAN_PARAM_GRID, results_dir=config.RESULTS_DIR)
print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
"
```
Expected: usa o cache parquet da C0 (sem rede se já baixado), imprime o dict padronizado de `in_sample_excellence` e cria `results/param_heatmap_*.png` + `results/in_sample_excellence_*.json`. Se a rede não estiver disponível e o cache não cobrir o intervalo, reportar e não travar.

- [ ] **Step 3: Inspecionar outputs**

Run: `ls -la results/ | tail -5`
Expected: o JSON `in_sample_excellence_*.json` e o `param_heatmap_*.png` presentes.

- [ ] **Step 4: Apresentar ao usuário**

Mostrar o dict (best_param, profit_factor, win_rate, n_trades, status) e o heatmap de parâmetros. Discutir se o resultado parece "colina suave" (robusto) ou "pico isolado" (suspeito) antes de iniciar o sub-ciclo 1B.

---

## Self-Review

- **Cobertura da spec:** engine VBT + smoke-test (T1), config/param_grid/thresholds (T2), `bar_returns`/`profit_factor`/`trade_stats` (T3–T5, spec §3/§6.2), Donchian long+short com canais e sem lookahead (T6, §6.1), `evaluate_grid` (T7, §6.3), `in_sample_excellence` + heatmap (T8, §6.4), `load_clean_ohlc` (T9, §6.5), critério de pronto + apresentação (T10, §10). Todos os itens do escopo 1A (§4) cobertos.
- **Refinamento declarado:** `trade_stats(signals, close)` (a spec §6.2 listou `trade_stats(signals)`; `win_rate` exige preços) — anotado na Task 5.
- **Placeholders:** nenhum — todo passo tem código/comando concreto.
- **Consistência de tipos:** `bar_returns(signals, close)`, `profit_factor(returns)`, `trade_stats(signals, close)`, `evaluate_grid(strategy_func, df, param_grid)`, `in_sample_excellence(strategy_func, df, param_grid, results_dir)` e `donchian(df, entry_lookback, exit_lookback)` usados de forma idêntica entre engine, layer1 e testes. Camada nomeada `in_sample_excellence` consistente entre `make_result` e os testes. Contrato `make_result`/`save_result` reusado da C0 sem alteração.
