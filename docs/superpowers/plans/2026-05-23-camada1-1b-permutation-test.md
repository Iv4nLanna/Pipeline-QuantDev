# Camada 1 — Sub-ciclo 1B (Passo 2: permutation_test_is) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar o Passo 2 da Camada 1 — `permutation_test_is` — que separa sinal real de ruído ajustado a dados, usando permutação OHLC com preservação de momentos estatísticos.

**Architecture:** `validation/permutation.py` (novo) hospeda `get_permutation`, ferramenta pura sobre `pd.DataFrame`/lista de DFs. `validation/layer1.py` ganha `permutation_test_is` + `_plot_pf_histogram`. Re-otimização completa do `param_grid` em cada uma das N permutações; p-valor Monte Carlo não-viesado = `(count(pf_perm >= pf_real) + 1) / (N + 1)`. Default `N=500`.

**Tech Stack:** Python 3.14, pandas, numpy, scipy.stats (skew, kurtosis), matplotlib, pytest. Spec: `docs/superpowers/specs/2026-05-23-camada1-1b-permutation-test.md`.

---

## File Structure

- `config.py` — + `N_PERMUTATIONS`, `P_VALUE_THRESHOLD`.
- `validation/permutation.py` — **novo**, `get_permutation(dfs, start_index=0, seed=None)`.
- `validation/layer1.py` — + `permutation_test_is`, + `_plot_pf_histogram`.
- `tests/test_permutation.py` — **novo**, 7 testes.
- `tests/test_layer1.py` — + 3 testes de `permutation_test_is`.

---

## Task 1: config.py — N_PERMUTATIONS e P_VALUE_THRESHOLD

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Anexar ao config.py**

```python
# Camada 1 — Passo 2 (permutation test)
N_PERMUTATIONS = 500             # default de desenvolvimento; 1000 em validação final
P_VALUE_THRESHOLD = 0.01         # p_value < 0.01 = aprovado no Passo 2
```

- [ ] **Step 2: Verificar import**

Run: `.venv/bin/python -c "import config; print(config.N_PERMUTATIONS, config.P_VALUE_THRESHOLD)"`
Expected: `500 0.01`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: config do Passo 2 (N_PERMUTATIONS, P_VALUE_THRESHOLD)"
```

---

## Task 2: get_permutation — núcleo (DataFrame único, sem start_index, com seed)

**Files:**
- Create: `validation/permutation.py`
- Test: `tests/test_permutation.py`

- [ ] **Step 1: Escrever os testes falhos (núcleo + propriedades estatísticas)**

```python
# tests/test_permutation.py
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
    # razão O/C, H/C, L/C invariante (apenas a escala multiplica tudo)
    np.testing.assert_allclose(perm["open"] / perm["close"],
                               df["open"] / df["close"], rtol=1e-9)
    np.testing.assert_allclose(perm["high"] / perm["close"],
                               df["high"] / df["close"], rtol=1e-9)
    np.testing.assert_allclose(perm["low"] / perm["close"],
                               df["low"] / df["close"], rtol=1e-9)
    # volume preservado
    np.testing.assert_array_equal(perm["volume"].values, df["volume"].values)
```

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_permutation.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'validation.permutation'`.

- [ ] **Step 3: Implementar get_permutation (núcleo)**

```python
# validation/permutation.py
import numpy as np
import pandas as pd


def _permute_single(df, perm_idx, start_index):
    """Aplica um vetor de permutação a um único DataFrame. perm_idx tem
    comprimento len(df) - 1 - start_index e indexa posições em [start_index+1, n-1]
    dos log-retornos."""
    n = len(df)
    log_close = np.log(df["close"].to_numpy())
    r = np.diff(log_close)                          # len n-1
    r_head = r[:start_index]                        # cabeçalho preservado
    r_tail_perm = r[start_index:][perm_idx]         # cauda embaralhada
    r_final = np.concatenate([r_head, r_tail_perm])
    close_perm = np.empty(n)
    close_perm[0] = df["close"].iloc[0]             # âncora
    close_perm[1:] = close_perm[0] * np.exp(np.cumsum(r_final))
    multiplier = close_perm / df["close"].to_numpy()
    out = df.copy()
    out["open"] = df["open"].to_numpy() * multiplier
    out["high"] = df["high"].to_numpy() * multiplier
    out["low"] = df["low"].to_numpy() * multiplier
    out["close"] = close_perm
    return out


def get_permutation(dfs, start_index=0, seed=None):
    """Embaralha log-retornos do close preservando momentos e estrutura OHLC.

    dfs: DataFrame único ou lista. Se lista, todos com o mesmo índice — o mesmo
         vetor de permutação é aplicado a todos (preserva correlação cruzada).
    start_index: barras [0, start_index] do close são intocadas; a permutação
                 atua nos log-retornos de índice >= start_index.
    seed: int opcional p/ reprodutibilidade.
    """
    is_list = isinstance(dfs, list)
    dfs_list = dfs if is_list else [dfs]
    if not dfs_list:
        raise ValueError("dfs vazio")
    n = len(dfs_list[0])
    for d in dfs_list[1:]:
        if len(d) != n or not d.index.equals(dfs_list[0].index):
            raise ValueError("DFs da lista devem ter o mesmo índice")
    if not (0 <= start_index <= n - 2):
        raise ValueError(f"start_index fora do range: {start_index} (n={n})")

    rng = np.random.default_rng(seed)
    tail_len = (n - 1) - start_index
    perm_idx = rng.permutation(tail_len)

    out = [_permute_single(d, perm_idx, start_index) for d in dfs_list]
    return out if is_list else out[0]
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_permutation.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/permutation.py tests/test_permutation.py
git commit -m "feat: get_permutation preserva momentos OHLC e destrói autocorrelação"
```

---

## Task 3: get_permutation — start_index

**Files:**
- Test: `tests/test_permutation.py` (anexar)

- [ ] **Step 1: Anexar teste**

```python
def test_get_permutation_start_index_preserva_cabecalho():
    df = _ar1(n=500)
    perm = get_permutation(df, start_index=100, seed=11)
    # close das primeiras 101 barras intacto (start_index permuta retornos a partir do índice 100)
    np.testing.assert_array_equal(perm["close"].iloc[:101].values,
                                  df["close"].iloc[:101].values)
    # restante alterado
    assert not np.array_equal(perm["close"].iloc[101:].values,
                              df["close"].iloc[101:].values)


def test_get_permutation_start_index_invalido_levanta():
    df = _ar1(n=50)
    with pytest.raises(ValueError):
        get_permutation(df, start_index=49)   # n-1 não cabe (precisa >= 2 elementos na cauda)
    with pytest.raises(ValueError):
        get_permutation(df, start_index=-1)
```

- [ ] **Step 2: Rodar**

Run: `.venv/bin/python -m pytest tests/test_permutation.py -k start_index -v`
Expected: 2 passed (lógica já implementada na Task 2).

- [ ] **Step 3: Commit**

```bash
git add tests/test_permutation.py
git commit -m "test: get_permutation com start_index e validação de range"
```

---

## Task 4: get_permutation — lista de DataFrames (preserva correlação cruzada)

**Files:**
- Test: `tests/test_permutation.py` (anexar)

- [ ] **Step 1: Anexar testes**

```python
def test_get_permutation_multi_df_mesmo_indice():
    df1 = _ar1(n=300, seed=1)
    df2 = _ar1(n=300, seed=2)
    df2.index = df1.index                              # garante mesmo índice
    p1, p2 = get_permutation([df1, df2], seed=5)
    # razão entre log-retornos preservada barra-a-barra:
    # se o mesmo perm_idx foi aplicado, a sequência de pares (r1[i], r2[i])
    # vem do mesmo i nos retornos originais.
    r1 = np.diff(np.log(df1["close"].to_numpy()))
    r2 = np.diff(np.log(df2["close"].to_numpy()))
    r1p = np.diff(np.log(p1["close"].to_numpy()))
    r2p = np.diff(np.log(p2["close"].to_numpy()))
    # Para cada barra na permutação, existe um i tal que (r1p[t], r2p[t]) == (r1[i], r2[i])
    pares_originais = set(zip(r1.round(12), r2.round(12)))
    pares_perm = set(zip(r1p.round(12), r2p.round(12)))
    assert pares_perm == pares_originais


def test_get_permutation_multi_df_indices_divergentes_levanta():
    df1 = _ar1(n=300)
    df2 = _ar1(n=200)
    with pytest.raises(ValueError):
        get_permutation([df1, df2], seed=0)
```

- [ ] **Step 2: Rodar**

Run: `.venv/bin/python -m pytest tests/test_permutation.py -k multi_df -v`
Expected: 2 passed (lógica já na Task 2).

- [ ] **Step 3: Commit**

```bash
git add tests/test_permutation.py
git commit -m "test: get_permutation aplica o mesmo perm_idx a múltiplos DFs"
```

---

## Task 5: permutation_test_is + _plot_pf_histogram (núcleo + viz)

**Files:**
- Modify: `validation/layer1.py`
- Test: `tests/test_layer1.py` (anexar)

> **Nota de design dos testes:** estratégias *não-path-dependentes* (ex.: `lambda df: Series(1.0)`) têm PF **invariante sob permutação** (PF depende do conjunto de retornos, não da ordem). Portanto não servem para testar o Passo 2. Os testes abaixo usam:
> (a) **Donchian real** sobre dados sintéticos com tendência forte (path-dependent → o PF cai sob permutação) — para o smoke-test de aprovação.
> (b) **monkeypatch de `evaluate_grid`** — para validar a matemática do p-valor com PFs controlados (determinístico, sem fragilidade estatística).
> (c) Helper `_df_tendencia` com `seed=0` fixo + `seed_base=0` no `permutation_test_is` para reprodutibilidade total.

- [ ] **Step 1: Anexar testes falhos (smoke + matemática + arquivos)**

```python
# tests/test_layer1.py (anexar)
import numpy as np
from validation.layer1 import permutation_test_is
from strategies.donchian import donchian
from validation.engine import evaluate_grid as _evaluate_grid_real


def _df_tendencia(n=500, drift=0.001, sigma=0.005, seed=0):
    """Série com tendência forte e ruído gaussiano (seed fixa = reprodutível)."""
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    rng = np.random.default_rng(seed)
    r = drift + rng.normal(0, sigma, n)
    close = 100.0 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1.0}, index=idx)


def test_permutation_test_is_estrutura_e_gates(tmp_path):
    """Smoke: dict padronizado + p_value em [0,1] + status binário, com Donchian em tendência."""
    df = _df_tendencia(n=400)
    grid = {"entry_lookback": [10, 20], "exit_lookback": [5, 10]}
    grid_real = _evaluate_grid_real(donchian, df, grid)
    best_idx = grid_real["profit_factor"].idxmax()
    best_param = {k: int(v) for k, v in zip(grid_real.index.names,
                  best_idx if isinstance(best_idx, tuple) else (best_idx,))}
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


def test_permutation_test_is_p_value_calculo_determinístico(tmp_path, monkeypatch):
    """Forçando PFs permutados conhecidos, p_value = count(>= pf_real) / N. Sem fragilidade."""
    df = _df_tendencia(n=100)
    pfs_perm_controlados = iter([0.5, 1.0, 1.5, 2.0, 2.5])  # 5 perms; 3 (>=1.5) extremos
    pf_real = 1.5
    # p_value = (3+1)/(5+1) = 4/6 ≈ 0.6667

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
    assert abs(r["metricas"]["p_value"] - 4/6) < 1e-9  # (3+1)/(5+1)


def test_permutation_test_is_salva_array_e_histograma(tmp_path):
    df = _df_tendencia(n=200)
    grid = {"entry_lookback": [10], "exit_lookback": [5]}
    grid_real = _evaluate_grid_real(donchian, df, grid)
    pf_real = float(grid_real["profit_factor"].max())
    best_param = {k: int(v) for k, v in zip(grid_real.index.names,
                  (grid_real["profit_factor"].idxmax(),)
                  if not isinstance(grid_real["profit_factor"].idxmax(), tuple)
                  else grid_real["profit_factor"].idxmax())}

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
    grid = {"a": [1], "b": [1]}

    def fake(df, **p):
        return pd.Series(0.0, index=df.index)

    with pytest.raises(ValueError):
        permutation_test_is(fake, df, grid, best_param={"a": 1, "b": 1},
                            pf_real=float("inf"), n_permutations=5,
                            results_dir=str(tmp_path))


import pytest  # importação local, no fim do bloco anexado, para não duplicar no topo do arquivo
```

> Se `pytest` já estiver importado no topo de `test_layer1.py`, remova o `import pytest` final.

- [ ] **Step 2: Rodar para confirmar falha**

Run: `.venv/bin/python -m pytest tests/test_layer1.py -k permutation_test_is -v`
Expected: FAIL com `ImportError: cannot import name 'permutation_test_is'`.

- [ ] **Step 3: Implementar `_plot_pf_histogram` + `permutation_test_is` (anexar em validation/layer1.py)**

```python
import math

from validation.permutation import get_permutation


def _plot_pf_histogram(pf_perm, pf_real, p_value, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pf_perm, bins=30, color="#4a7", alpha=0.7, edgecolor="black")
    ax.axvline(pf_real, color="red", linestyle="--", lw=2,
               label=f"pf_real={pf_real:.3f}")
    ax.set_xlabel("profit_factor (permutação)")
    ax.set_ylabel("frequência")
    ax.set_title(f"Distribuição nula vs. PF real — p-valor={p_value:.4f} "
                 f"({len(pf_perm)} permutações)")
    ax.legend(loc="upper right")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = os.path.join(results_dir, f"pf_perm_histograma_{ts}.png")
    fig.savefig(caminho, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return caminho


def permutation_test_is(strategy_func, df, param_grid, best_param,
                        pf_real, n_permutations=None, seed_base=0,
                        results_dir="results"):
    """Passo 2 da Camada 1: teste de permutação in-sample.

    Para cada permutação, re-otimiza param_grid e usa o MAX PF como estatística.
    p_value = count(pf_perm >= pf_real) / n_permutations.
    """
    if n_permutations is None:
        n_permutations = config.N_PERMUTATIONS
    if not math.isfinite(pf_real):
        raise ValueError("pf_real infinito; rever red-flag REDFLAG_MAX_PF")

    pf_perm = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        df_perm = get_permutation(df, start_index=0, seed=seed_base + i)
        grid_df = evaluate_grid(strategy_func, df_perm, param_grid)
        pf_finito = grid_df["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
        pf_perm[i] = float(pf_finito.max()) if not pf_finito.empty else 0.0

    n_extremos = int((pf_perm >= pf_real).sum())
    p_value = (n_extremos + 1) / (n_permutations + 1)

    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    array_path = os.path.join(results_dir, f"pf_perm_{ts}.npy")
    np.save(array_path, pf_perm)
    histograma = _plot_pf_histogram(pf_perm, pf_real, p_value, results_dir)

    metricas = {
        "best_param": best_param,
        "profit_factor_real": float(pf_real),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "pf_perm_array_path": array_path,
        "histograma": histograma,
    }
    if p_value < config.P_VALUE_THRESHOLD:
        status = "aprovado"
        motivo = f"p_value={p_value:.4f} < {config.P_VALUE_THRESHOLD} em {n_permutations} permutações."
        proximo = "Avançar para o Passo 3 (walk_forward_test)."
    else:
        status = "reprovado"
        motivo = f"p_value={p_value:.4f} >= {config.P_VALUE_THRESHOLD} em {n_permutations} permutações."
        proximo = "Estratégia indistinguível de ruído otimizado; não avançar."

    resultado = make_result("permutation_test_is", status, metricas, motivo, proximo)
    save_result(resultado, results_dir=results_dir)
    return resultado
```

- [ ] **Step 4: Rodar para confirmar passagem**

Run: `.venv/bin/python -m pytest tests/test_layer1.py -v`
Expected: todos os testes de layer1 passam (3 do Passo 1 + 4 do Passo 2 = 7).

- [ ] **Step 5: Commit**

```bash
git add validation/layer1.py tests/test_layer1.py
git commit -m "feat: permutation_test_is (Passo 2) com p-valor, histograma e gate"
```

---

## Task 6: Execução real do Passo 2 sobre BTC (continuação do Passo 1)

**Files:** nenhum arquivo novo (execução + apresentação).

- [ ] **Step 1: Rodar a suíte inteira**

Run: `.venv/bin/python -m pytest -q`
Expected: tudo passa. Confirma que 1B não regrediu 1A.

- [ ] **Step 2: Executar Passo 1 + Passo 2 encadeados com BTC real**

Run:
```bash
.venv/bin/python -c "
import config, json
from data.layer0 import load_clean_ohlc
from strategies.donchian import donchian
from validation.layer1 import in_sample_excellence, permutation_test_is

df = load_clean_ohlc(config.TICKER, config.TIMEFRAME, config.START, config.END,
                     gap_threshold=config.GAP_THRESHOLD, cache_dir=config.CACHE_DIR,
                     exchange_name=config.EXCHANGE)
print(f'>> dataset: {len(df)} barras')

r1 = in_sample_excellence(donchian, df, config.DONCHIAN_PARAM_GRID,
                          results_dir=config.RESULTS_DIR)
print('--- PASSO 1 ---')
print(json.dumps(r1, indent=2, ensure_ascii=False, default=str))

assert r1['status'] == 'aprovado', 'Passo 1 reprovado — não rodar Passo 2'

r2 = permutation_test_is(donchian, df, config.DONCHIAN_PARAM_GRID,
                         best_param=r1['metricas']['best_param'],
                         pf_real=r1['metricas']['profit_factor'],
                         n_permutations=config.N_PERMUTATIONS,
                         seed_base=0,
                         results_dir=config.RESULTS_DIR)
print('--- PASSO 2 ---')
print(json.dumps(r2, indent=2, ensure_ascii=False, default=str))
"
```
Expected: o dict do Passo 1 (aprovado, conforme execução anterior) seguido do dict do Passo 2 com `p_value`, `histograma` e `pf_perm_array_path`. Tempo esperado para 500 permutações × 24 combos × 55k barras: avaliar; se > 5 min, parar e discutir aceleração (não improvisar).

- [ ] **Step 3: Inspecionar outputs**

Run: `ls -la results/ | tail -5`
Expected: `pf_perm_<ts>.npy`, `pf_perm_histograma_<ts>.png`, `permutation_test_is_<ts>.json` presentes.

- [ ] **Step 4: Apresentar ao usuário**

Mostrar:
- O dict do Passo 2 (best_param, pf_real, p_value, n_permutations, status).
- O histograma PNG (linha vermelha do PF real sobre a distribuição nula).
- Leitura: PF real cai na cauda direita (sinal real) ou na massa central (ruído otimizado)?

---

## Self-Review

- **Cobertura da spec (§2.1–§2.5):**
  - §2.1 (herança da convenção) — `permutation_test_is` chama `evaluate_grid` que chama `bar_returns`; nenhuma redefinição (Task 5).
  - §2.2 (algoritmo de permutação) — Tasks 2–4 implementam e testam preservação de momentos, destruição de autocorrelação, cabeçalho intacto, multi-DF, reprodutibilidade por seed.
  - §2.3 (p-valor) — Task 5 implementa exatamente `count / n_perms`; teste determinístico `test_permutation_test_is_p_value_calculo_determinístico` amarra o cálculo via `monkeypatch` (não depende de RNG).
  - §2.4 (config) — Task 1 expõe `N_PERMUTATIONS=500` e `P_VALUE_THRESHOLD=0.01`.
  - §2.5 (output: dict + array .npy + histograma) — Task 5 cobre tudo numa única implementação coesa.
- **Placeholders:** nenhum. O histograma e a função principal nascem juntos na Task 5 (era um stub-bridge na versão anterior do plano; eliminado).
- **Robustez dos testes:** os tests do Passo 2 evitam o pitfall da invariância de PF sob permutação (estratégia não-path-dependent → PF idêntico em todas as perms → p_value=1). O smoke-test usa `donchian` real (path-dependent); o teste de matemática do p-valor usa monkeypatch de `evaluate_grid` para resultados determinísticos.
- **Consistência de tipos:** `get_permutation(dfs, start_index, seed) -> DF | List[DF]` usado de forma idêntica entre `validation/permutation.py`, testes e `permutation_test_is`. `permutation_test_is(strategy_func, df, param_grid, best_param, pf_real, n_permutations, seed_base, results_dir)` único entre layer1 e testes. `make_result("permutation_test_is", ...)` consistente com `{aprovado, reprovado, revisar}` de `utils/contract.py`.
- **Ausência de execução prematura:** a Task 6 (execução real) só roda depois de TODOS os testes passarem (Step 1). O Step 2 contém o guard `assert r1['status'] == 'aprovado'`, impedindo avançar se algo regrediu em 1A.
